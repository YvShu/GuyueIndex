/*
 * @Author: Guyue
 * @Date: 2026-03-23 15:36:10
 * @LastEditTime: 2026-03-23 15:39:02
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/reindexing_manager.cpp
 */
#include <reindexing_manager.h>

ReindexingManager::ReindexingManager(std::shared_ptr<ReindexingParams> reindexing_params)
{
    if (reindexing_params->reindexing_strategy == "DeDrift")
    {

    } else if (reindexing_params->reindexing_strategy == "LIRE") {

    } else if (reindexing_params->reindexing_strategy == "AdaIVF") {

    } else {
        throw std::runtime_error("[ReindexingManager] ReindexingManager: The maintenance strategy does not exist.");
    }
}

ReindexingManager::~ReindexingManager() {}