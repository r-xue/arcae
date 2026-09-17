#ifndef ARCAE_CELL_SLICE_H
#define ARCAE_CELL_SLICE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>

#include "arcae/isolated_table_proxy.h"

namespace arcae {
namespace detail {

arrow::Future<std::shared_ptr<arrow::Array>> ReadCellSliceImpl(
    const std::shared_ptr<IsolatedTableProxy>& itp, const std::string& column,
    int64_t rownr, const std::vector<int64_t>& blc,
    const std::vector<int64_t>& trc, const std::vector<int64_t>& inc = {});

}  // namespace detail
}  // namespace arcae

#endif  // ARCAE_CELL_SLICE_H
