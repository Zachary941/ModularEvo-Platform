import api from './index'

/**
 * 第四章 API 封装 — AutoRouter
 */

// 获取模型加载状态
export const getStatus = () => api.get('/ch4/status')

// 获取基线准确率
export const getBaseline = () => api.get('/ch4/baseline')

// 获取所有模型信息
export const getModels = () => api.get('/ch4/models')

// 上传数据集
export const uploadDataset = (file) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/ch4/upload-dataset', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

// 下载示例数据集
export const downloadSampleDataset = () =>
  api.get('/ch4/sample-dataset', { responseType: 'blob' })

// 启动评测
export const evaluate = (filePath) =>
  api.post('/ch4/evaluate', { file_path: filePath })

// 获取知识图谱数据
export const getGraph = () => api.get('/ch4/graph')
