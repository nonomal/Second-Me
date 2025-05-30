import { create } from 'zustand';
import { getModelConfig, updateModelConfig } from '../service/modelConfig';

export interface CloudProviderConfig {
  provider_type: string;
  cloud_service_api_key?: string;
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

export const useCloudProviderStore = create<CloudProviderState>((set, get) => ({
  cloudConfig: {
    provider_type: '',
    cloud_service_api_key: '',
    api_endpoint: '',
    model_name: '',
    region: ''
  },
  updateCloudConfig: (config) => set({ cloudConfig: config }),
  saveCloudConfig: async () => {
    const { cloudConfig } = get();
    try {
      const response = await getModelConfig();
      if (response.data.data) {
        const currentConfig = response.data.data;

        await updateModelConfig({
          ...currentConfig,
          cloud_service_api_key: cloudConfig.cloud_service_api_key || ''
        });
      }
    } catch (error) {
      console.error('Failed to save cloud config:', error);

      throw error;
    }
  },
  fetchCloudConfig: async () => {
    try {
      const response = await getModelConfig();
      if (response.data.data && response.data.data.cloud_service_api_key) {
        set({
          cloudConfig: {
            provider_type: response.data.data.cloud_service_api_key ? 'alibaba' : '',
            cloud_service_api_key: response.data.data.cloud_service_api_key,
            api_endpoint: '',
            model_name: '',
            region: ''
          }
        });
      }
    } catch (error) {
      console.error('Failed to fetch cloud config:', error);
      throw error;
    }
  }
}));
