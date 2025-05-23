import { create } from 'zustand';

export interface CloudProviderConfig {
  provider_type: string;
  api_key?: string;
  api_endpoint?: string;
  model_name?: string;
  region?: string;
}

interface CloudProviderState {
  cloudConfig: CloudProviderConfig;
  updateCloudConfig: (config: CloudProviderConfig) => void;
  saveCloudConfig: () => Promise<void>;
  fetchCloudConfig: () => Promise<void>;
}

export const useCloudProviderStore = create<CloudProviderState>((set) => ({
  cloudConfig: {
    provider_type: '',
    api_key: '',
    api_endpoint: '',
    model_name: '',
    region: ''
  },
  updateCloudConfig: (config) => set({ cloudConfig: config }),
  saveCloudConfig: async () => {
    // 这里应该实现保存配置到后端的逻辑
    // 目前只是模拟成功
    return Promise.resolve();
  },
  fetchCloudConfig: async () => {
    // 这里应该实现从后端获取配置的逻辑
    // 目前只是返回当前状态
    return Promise.resolve();
  }
}));
