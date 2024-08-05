#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }
    // 内存空间并没有分配，而是在调用getPtr函数的时候，直接根据最大的内存分配的
    // 所以alloc和free每个函数开头都要断言指针是空的，因为内存压根没有分配

    //@param size: the size of memory to be allocated
    //@function: alloc memory from runtime
    //@return: the starting address of the allocated memory
    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        size_t headAddr = this->peak;
        freeBlockInfo tmp = {(size_t)0, size};
        auto it = this->freeBlocks.lower_bound(tmp);

        if (it != this->freeBlocks.end()) {
            size_t blockSize = it->blockSize;
            headAddr = it->addr;
            size_t tailAddr = headAddr + size;

            // 更新头尾hashmap
            this->headAddrMap.erase(headAddr);
            this->tailAddrMap.erase(tailAddr);

            if(blockSize>size) { //这个空闲块还有剩下的部分
                freeBlockInfo newBlockInfo{headAddr+size, blockSize-size};
                this->freeBlocks.insert(newBlockInfo);
                this->headAddrMap[headAddr+size]=newBlockInfo.blockSize;
                this->tailAddrMap[headAddr+blockSize]=newBlockInfo.blockSize;
            }
            this->freeBlocks.erase(it);
            this->used += size; 
        } else {
            auto end = this->tailAddrMap.find(this->peak);
            if(end!=this->tailAddrMap.end()) {
                headAddr = this->peak-end->second;
                IT_ASSERT(end->second<size);
                this->peak = (size-end->second);

                freeBlockInfo endBlockInfo{headAddr,end->second};
                this->freeBlocks.insert(endBlockInfo);
                this->headAddrMap.erase(endBlockInfo.addr);
                this->tailAddrMap.erase(endBlockInfo.addr+endBlockInfo.blockSize);
            } else {
                this->peak += size;
            }
            this->used += size;
        }

        return headAddr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t tailAddr = addr + size;
        freeBlockInfo block={addr,size};
        this->headAddrMap[addr]=block.blockSize;
        this->tailAddrMap[tailAddr]=block.blockSize;
        
        //合并相邻的空闲块
        auto pre = this->tailAddrMap.find(addr);
        auto next = this->headAddrMap.find(tailAddr);
        if(pre!=this->tailAddrMap.end()) {
            size_t preSize = pre->second;
            this->headAddrMap.erase(block.addr);
            this->headAddrMap[block.addr-preSize] += block.blockSize;
            this->tailAddrMap.erase(block.addr);
            this->tailAddrMap[tailAddr] += preSize;
            block.addr -= preSize;
            block.blockSize += preSize;
            this->freeBlocks.erase(freeBlockInfo{block.addr,preSize});
        }

        if(next!=this->headAddrMap.end()) {
            auto nextSize = next->second;
            this->headAddrMap.erase(tailAddr);
            this->headAddrMap[block.addr] += nextSize;
            this->tailAddrMap.erase(tailAddr);
            this->tailAddrMap[tailAddr+nextSize] += block.blockSize;

            tailAddr += nextSize;
            block.blockSize += nextSize;
            this->freeBlocks.erase(freeBlockInfo{tailAddr-nextSize,nextSize});
        }
        this->freeBlocks.insert(block);
        this->used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size) const
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info() const
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
