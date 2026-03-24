/*
 * @Author: Guyue
 * @Date: 2026-03-23 15:27:43
 * @LastEditTime: 2026-03-23 15:35:56
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/reindexing_manager.h
 */
#ifndef REINDEXING_H
#define REINDEXING_H

#include <partition_manager.h>
#include <memory>
#include <common.h>

class ReindexingManager 
{
    std::shared_ptr<PartitionManager> centroids_manager_; // 分区中心管理器
    std::shared_ptr<PartitionManager> partition_manager_; // 分区管理器
    std::shared_ptr<ReindexingParams> reindexing_params_; // 索引维护参数

    /**
     * @brief: 索引维护器构造函数
     * @param {shared_ptr<ReindexingParams>} reindexing_params 索引维护参数
     * @return {*}
     */    
    ReindexingManager(std::shared_ptr<ReindexingParams> reindexing_params);

    /**
     * @brief: 索引维护器析构函数
     * @return {*}
     */    
    ~ReindexingManager();
}; 

#endif // ReindexingManager