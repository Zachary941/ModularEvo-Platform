import api from './index'

/**
 * 第三章 API 封装
 */

// 获取可用模块列表
export const getModules = () => api.get('/ch3/modules')

// 加载模块 (返回稀疏率统计)
export const loadModule = (language) =>
  api.post('/ch3/load-module', { language })

// 获取微调模型列表
export const getFinetuned = () => api.get('/ch3/finetuned')

// 评测微调模型
export const evaluateFinetuned = (task) =>
  api.post(`/ch3/evaluate/${task}`)

// 执行模型合并 + 评测
export const mergeModels = (params) =>
  api.post('/ch3/merge', params)

// 获取知识图谱数据
export const getGraph = () => api.get('/ch3/graph')
