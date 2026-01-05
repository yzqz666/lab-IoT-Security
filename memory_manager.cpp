#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>

/*
 * 操作系统课程设计：内存管理模拟系统设计
 *
 * 该程序实现动态分区内存管理，支持首次适应（First Fit）、最佳适应（Best Fit）、
 * 最坏适应（Worst Fit）等三种分配算法。用户可以初始化总内存大小，申请内存、
 * 回收内存、进行内存紧凑、查看内存状态以及查看内存利用率和外部碎片率等统计信息。
 */

struct Partition {
    int id;            // 分区标识符（进程ID或者分区编号）
    size_t start;      // 起始地址
    size_t size;       // 分区大小
    bool allocated;    // 是否已分配
    std::string name;  // 进程名称，用于显示
};

class MemoryManager {
public:
    MemoryManager(size_t totalSize) : totalSize(totalSize), nextId(1) {
        Partition initial{0, 0, totalSize, false, ""};
        partitions.push_back(initial);
    }

    // 显示当前内存分区状态
    void display(){
    std::cout << "\nMemory Partition Status:\n";
    std::cout << std::left << std::setw(10) << "ID"
              << std::setw(15) << "Start"
              << std::setw(15) << "Size"
              << std::setw(15) << "Status" 
              << std::setw(10) << "Process" << '\n';
    for (const auto &p : partitions) {
        std::cout << std::left << std::setw(10) << p.id
                  << std::setw(15) << p.start
                  << std::setw(15) << p.size
                  << std::setw(15) << (p.allocated ? "Allocated" : "Free")
                  << std::setw(10) << (p.allocated ? p.name : "")
                  << '\n';
    }
}

    // 显示统计信息：内存利用率和外部碎片率
    void showStatistics() const {
        size_t used = 0;
        size_t freeMemory = 0;
        size_t freeBlocks = 0;
        for (const auto &p : partitions) {
            if (p.allocated)
                used += p.size;
            else {
                freeMemory += p.size;
                freeBlocks++;
            }
        }
        double utilization = totalSize ? (double)used / totalSize : 0.0;
        double externalFragmentation = totalSize ? (double)freeBlocks / partitions.size() : 0.0;
        std::cout << "\n统计信息:\n";
        std::cout << "总内存大小: " << totalSize << '\n';
        std::cout << "已用内存: " << used << '\n';
        std::cout << "空闲内存: " << freeMemory << '\n';
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "内存利用率: " << utilization * 100 << "%\n";
        std::cout << "外部碎片率（按分区数量占比）: " << externalFragmentation * 100 << "%\n";
    }

    // 首次适应算法
    bool firstFit(const std::string &name, size_t size) {
        for (size_t i = 0; i < partitions.size(); ++i) {
            if (!partitions[i].allocated && partitions[i].size >= size) {
                allocateBlock(i, name, size);
                return true;
            }
        }
        return false;
    }

    // 最佳适应算法
    bool bestFit(const std::string &name, size_t size) {
        // 找到最小但足够的空闲分区
        size_t bestIndex = partitions.size();
        size_t bestSize = static_cast<size_t>(-1);
        for (size_t i = 0; i < partitions.size(); ++i) {
            if (!partitions[i].allocated && partitions[i].size >= size && partitions[i].size < bestSize) {
                bestSize = partitions[i].size;
                bestIndex = i;
            }
        }
        if (bestIndex < partitions.size()) {
            allocateBlock(bestIndex, name, size);
            return true;
        }
        return false;
    }

    // 最坏适应算法
    bool worstFit(const std::string &name, size_t size) {
        // 找到最大的空闲分区
        size_t worstIndex = partitions.size();
        size_t worstSize = 0;
        for (size_t i = 0; i < partitions.size(); ++i) {
            if (!partitions[i].allocated && partitions[i].size >= size && partitions[i].size > worstSize) {
                worstSize = partitions[i].size;
                worstIndex = i;
            }
        }
        if (worstIndex < partitions.size()) {
            allocateBlock(worstIndex, name, size);
            return true;
        }
        return false;
    }

    // 释放内存
    bool freeMemory(int id) {
        for (size_t i = 0; i < partitions.size(); ++i) {
            if (partitions[i].allocated && partitions[i].id == id) {
                partitions[i].allocated = false;
                partitions[i].name.clear();
                // 合并相邻空闲分区
                mergeAdjacent();
                return true;
            }
        }
        return false;
    }

    // 内存紧凑：将所有已分配分区移动到内存低地址端，把所有空闲分区合并为一个
    void compact() {
        size_t currentStart = 0;
        std::vector<Partition> newPartitions;
        // 先移动所有已分配分区，重新计算起始地址
        for (auto &p : partitions) {
            if (p.allocated) {
                Partition moved = p;
                moved.start = currentStart;
                currentStart += moved.size;
                newPartitions.push_back(moved);
            }
        }
        // 计算剩余空闲大小
        size_t freeSize = totalSize - currentStart;
        if (freeSize > 0) {
            Partition freeBlock;
            freeBlock.id = 0;
            freeBlock.start = currentStart;
            freeBlock.size = freeSize;
            freeBlock.allocated = false;
            newPartitions.push_back(freeBlock);
        }
        partitions = std::move(newPartitions);
    }

private:
    size_t totalSize;               // 总内存大小
    std::vector<Partition> partitions; // 内存分区列表
    int nextId;                     // 用于生成唯一ID

    // 在指定索引处分配块，按请求大小拆分分区
    void allocateBlock(size_t index, const std::string &name, size_t size) {
        Partition &p = partitions[index];
        Partition allocatedPart;
        allocatedPart.id = nextId++;
        allocatedPart.start = p.start;
        allocatedPart.size = size;
        allocatedPart.allocated = true;
        allocatedPart.name = name;
        // 如果分区正好大小
        if (p.size == size) {
            partitions[index] = allocatedPart;
        } else {
            // 将原分区拆分为已分配和剩余空闲两部分
            Partition freePart;
            freePart.id = 0;
            freePart.start = p.start + size;
            freePart.size = p.size - size;
            freePart.allocated = false;
            freePart.name.clear();
            // 替换当前索引为已分配分区
            partitions[index] = allocatedPart;
            // 在后面插入空闲分区
            partitions.insert(partitions.begin() + index + 1, freePart);
        }
    }

    // 合并相邻空闲分区
    void mergeAdjacent() {
        for (size_t i = 0; i + 1 < partitions.size();) {
            if (!partitions[i].allocated && !partitions[i + 1].allocated) {
                partitions[i].size += partitions[i + 1].size;
                partitions.erase(partitions.begin() + i + 1);
                // 不增加i，继续检查合并后的分区与下一个分区
            } else {
                ++i;
            }
        }
    }
};

// 显示菜单
void showMenu() {
    std::cout << "\n=== 内存管理模拟系统 ===\n";
    std::cout << "1. 显示内存状态\n";
    std::cout << "2. 分配内存\n";
    std::cout << "3. 释放内存\n";
    std::cout << "4. 内存紧凑\n";
    std::cout << "5. 统计信息\n";
    std::cout << "6. 退出\n";
    std::cout << "请选择操作: ";
}

int main() {
    std::cout << "请输入总内存大小（字节或KB的整数值）: ";
    size_t totalSize;
    std::cin >> totalSize;
    MemoryManager manager(totalSize);
    bool running = true;
    while (running) {
        showMenu();
        int choice;
        std::cin >> choice;
        switch (choice) {
        case 1: {
            manager.display();
            break;
        }
        case 2: {
            std::cout << "请输入进程名称: ";
            std::string name;
            std::cin >> name;
            std::cout << "请输入申请的内存大小: ";
            size_t size;
            std::cin >> size;
            std::cout << "请选择分配算法(1-首次适应, 2-最佳适应, 3-最坏适应): ";
            int algo;
            std::cin >> algo;
            bool success = false;
            switch (algo) {
            case 1:
                success = manager.firstFit(name, size);
                break;
            case 2:
                success = manager.bestFit(name, size);
                break;
            case 3:
                success = manager.worstFit(name, size);
                break;
            default:
                std::cout << "无效算法选择\n";
                break;
            }
            if (success) {
                std::cout << "分配成功。\n";
            } else {
                std::cout << "分配失败，内存不足。\n";
            }
            break;
        }
        case 3: {
            std::cout << "请输入要释放的分区ID: ";
            int id;
            std::cin >> id;
            bool result = manager.freeMemory(id);
            if (result) {
                std::cout << "释放成功。\n";
            } else {
                std::cout << "未找到对应分区或分区未分配。\n";
            }
            break;
        }
        case 4: {
            manager.compact();
            std::cout << "已完成内存紧凑。\n";
            break;
        }
        case 5: {
            manager.showStatistics();
            break;
        }
        case 6: {
            running = false;
            break;
        }
        default:
            std::cout << "无效选择，请重新输入。\n";
            break;
        }
    }
    std::cout << "退出程序。\n";
    return 0;
}