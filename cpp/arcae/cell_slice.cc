#include "arcae/cell_slice.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Slicer.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ColumnDesc.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableColumn.h>
#include <casacore/tables/Tables/TableProxy.h>

#include "arcae/table_utils.h"
#include "arcae/type_traits.h"

using ::arrow::Array;
using ::arrow::ArrayData;
using ::arrow::Buffer;
using ::arrow::FixedSizeListArray;
using ::arrow::Future;
using ::arrow::Result;
using ::arrow::Status;
using ::arrow::StringBuilder;

template <class CT>
using CasaArray = ::casacore::Array<CT>;
using ::casacore::ArrayColumn;
using ::casacore::DataType;
using ::casacore::IPosition;
using ::casacore::Slicer;
using ::casacore::Table;
using ::casacore::TableColumn;
using ::casacore::TableProxy;

namespace arcae {
namespace detail {

namespace {

// Create a primitive array from an allocated buffer
std::shared_ptr<Array> MakePrimitiveArray(std::shared_ptr<arrow::DataType> dtype,
                                          const std::int64_t N,
                                          std::shared_ptr<Buffer> buffer) {
  auto array_data = std::make_shared<ArrayData>(
      std::move(dtype), N, std::vector<std::shared_ptr<Buffer>>{nullptr, buffer});
  return arrow::MakeArray(array_data);
}

template <DataType CDT>
Result<std::shared_ptr<Array>> ReadTypedCellSlice(
    const Table& table, const std::string& column, int64_t rownr,
    const Slicer& slicer, const IPosition& slice_shape) {
  using CT = typename CasaDataTypeTraits<CDT>::CasaType;
  constexpr bool is_complex = CasaDataTypeTraits<CDT>::is_complex;
  int64_t nelements = slice_shape.product();

  ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateBuffer(nelements * sizeof(CT)));
  CasaArray<CT> arr(slice_shape, buffer->template mutable_data_as<CT>(),
                    casacore::SHARE);
  ArrayColumn<CT> array_col(table, column);
  array_col.getSlice(rownr, slicer, arr);

  auto arrow_dtype = CasaDataTypeTraits<CDT>::ArrowDataType();
  if (!arrow_dtype) {
    return Status::Invalid("Unsupported Arrow data type for CASA type ", CDT);
  }
  int64_t prim_len = is_complex ? 2 * nelements : nelements;
  std::shared_ptr<Array> result =
      MakePrimitiveArray(std::move(arrow_dtype), prim_len, std::move(buffer));

  if constexpr (is_complex) {
    ARROW_ASSIGN_OR_RAISE(result, FixedSizeListArray::FromArrays(result, 2));
  }

  for (int d = 0; d < static_cast<int>(slice_shape.size()) - 1; ++d) {
    ARROW_ASSIGN_OR_RAISE(result,
                          FixedSizeListArray::FromArrays(result, slice_shape[d]));
  }

  return result;
}

Result<std::shared_ptr<Array>> ReadStringCellSlice(
    const Table& table, const std::string& column, int64_t rownr,
    const Slicer& slicer, const IPosition& slice_shape) {
  CasaArray<casacore::String> arr(slice_shape);
  ArrayColumn<casacore::String> array_col(table, column);
  array_col.getSlice(rownr, slicer, arr);

  int64_t nelements = slice_shape.product();
  StringBuilder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(nelements));
  for (const auto& str : arr) {
    ARROW_RETURN_NOT_OK(builder.Append(str));
  }
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Array> result, builder.Finish());

  for (int d = 0; d < static_cast<int>(slice_shape.size()) - 1; ++d) {
    ARROW_ASSIGN_OR_RAISE(result,
                          FixedSizeListArray::FromArrays(result, slice_shape[d]));
  }
  return result;
}

}  // namespace

Future<std::shared_ptr<Array>> ReadCellSliceImpl(
    const std::shared_ptr<IsolatedTableProxy>& itp, const std::string& column,
    int64_t rownr, const std::vector<int64_t>& blc,
    const std::vector<int64_t>& trc, const std::vector<int64_t>& inc) {
  return itp->RunAsync(
      [column = column, rownr, blc = blc, trc = trc, inc = inc](
          const TableProxy& tp) -> Result<std::shared_ptr<Array>> {
        ARROW_RETURN_NOT_OK(ColumnExists(tp.table(), column));
        TableColumn table_col(tp.table(), column);
        const auto& col_desc = table_col.columnDesc();

        if (!col_desc.isArray()) {
          return Status::Invalid("Column '", column, "' is not an array column");
        }

        if (rownr < 0 || rownr >= static_cast<int64_t>(tp.table().nrow())) {
          return Status::IndexError("Row ", rownr, " is out of bounds [0, ",
                                    tp.table().nrow(), ")");
        }

        if (!table_col.isDefined(rownr)) {
          return Status::Invalid("Row ", rownr, " in column '", column,
                                 "' is not defined");
        }

        IPosition cell_shape = table_col.shape(rownr);
        size_t ndim = cell_shape.size();

        if (blc.size() != ndim || trc.size() != ndim) {
          return Status::Invalid("blc/trc dimensions (", blc.size(), "/",
                                 trc.size(), ") do not match cell ndim (", ndim,
                                 ")");
        }

        if (!inc.empty() && inc.size() != ndim) {
          return Status::Invalid("inc dimension (", inc.size(),
                                 ") does not match cell ndim (", ndim, ")");
        }

        IPosition f_start(ndim, 0);
        IPosition f_end(ndim, 0);
        IPosition f_inc(ndim, 1);

        for (size_t i = 0; i < ndim; ++i) {
          size_t c_idx = ndim - 1 - i;
          f_start[i] = blc[c_idx];
          f_end[i] = trc[c_idx];
          f_inc[i] = inc.empty() ? 1 : inc[c_idx];

          if (f_start[i] < 0 || f_start[i] >= cell_shape[i]) {
            return Status::IndexError(
                "blc index ", f_start[i], " exceeds dimension ", c_idx, " (size ",
                cell_shape[i], ") in column '", column, "'");
          }
          if (f_end[i] < f_start[i] || f_end[i] >= cell_shape[i]) {
            return Status::IndexError(
                "trc index ", f_end[i], " is invalid for dimension ", c_idx,
                " with blc ", f_start[i], " (size ", cell_shape[i],
                ") in column '", column, "'");
          }
          if (f_inc[i] <= 0) {
            return Status::Invalid("inc must be positive in dimension ", c_idx);
          }
        }

        Slicer slicer(f_start, f_end, f_inc, Slicer::endIsLast);
        IPosition slice_shape = slicer.length();

        auto dtype = col_desc.dataType();
        switch (dtype) {
          case DataType::TpBool:
            return ReadTypedCellSlice<DataType::TpBool>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpChar:
            return ReadTypedCellSlice<DataType::TpChar>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpUChar:
            return ReadTypedCellSlice<DataType::TpUChar>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpShort:
            return ReadTypedCellSlice<DataType::TpShort>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpUShort:
            return ReadTypedCellSlice<DataType::TpUShort>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpInt:
            return ReadTypedCellSlice<DataType::TpInt>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpUInt:
            return ReadTypedCellSlice<DataType::TpUInt>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpInt64:
            return ReadTypedCellSlice<DataType::TpInt64>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpFloat:
            return ReadTypedCellSlice<DataType::TpFloat>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpDouble:
            return ReadTypedCellSlice<DataType::TpDouble>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpComplex:
            return ReadTypedCellSlice<DataType::TpComplex>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpDComplex:
            return ReadTypedCellSlice<DataType::TpDComplex>(
                tp.table(), column, rownr, slicer, slice_shape);
          case DataType::TpString:
            return ReadStringCellSlice(tp.table(), column, rownr, slicer,
                                       slice_shape);
          default:
            return Status::NotImplemented("Cell slice for CASA data type ",
                                          dtype);
        }
      });
}

}  // namespace detail
}  // namespace arcae
