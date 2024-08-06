#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        // 根据perm来调换顺序
        vector<Shape> shapeVec;
        for(auto &tensor:inputs) {
            auto input_dim=tensor->getDims();
            int rank=tensor->getRank();
            Shape output_dim(rank);
            auto perm=this->getPermute();
            for(int i=0;i<rank;++i) {
                output_dim[i]=input_dim[perm[i]];
            }
            shapeVec.push_back(output_dim);
        }
        // =================================== 作业 ===================================
        // TODO：修改 output_dim，返回正确的 transpose 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
        // =================================== 作业 ===================================


        return {shapeVec};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
