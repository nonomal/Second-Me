import { Modal, Radio, Input, message } from 'antd';
import Image from 'next/image';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useCallback, useEffect, useState, useRef } from 'react';
import { updateModelConfig, getModelConfig } from '../../service/modelConfig';

interface CloudProviderConfig {
  provider_type: string;
  cloud_service_api_key?: string;
}

interface IProps {
  open: boolean;
  onClose: () => void;
  cloudConfig: CloudProviderConfig;
  updateCloudConfig: (config: CloudProviderConfig) => void;
  saveCloudConfig: () => Promise<void>;
}

const options = [
  {
    label: 'None',
    value: ''
  },
  {
    label: 'Model Studio',
    value: 'alibaba'
  }
];

const CloudProviderModal = (props: IProps): JSX.Element => {
  const { open, onClose, cloudConfig, updateCloudConfig, saveCloudConfig } = props;
  const [providerType, setProviderType] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [originalConfig, setOriginalConfig] = useState<CloudProviderConfig | null>(null);
  const [savedApiKey, setSavedApiKey] = useState<string>('');

  const configRef = useRef(cloudConfig);
  const updateConfigRef = useRef(updateCloudConfig);

  useEffect(() => {
    configRef.current = cloudConfig;
    updateConfigRef.current = updateCloudConfig;
  }, [cloudConfig, updateCloudConfig]);

  // 预加载 API key
  useEffect(() => {
    const loadApiKey = async () => {
      try {
        const res = await getModelConfig();
        if (res.data?.data?.cloud_service_api_key) {
          setSavedApiKey(res.data.data.cloud_service_api_key);
        }
      } catch (error) {
        console.error('Failed to load API key:', error);
      }
    };

    loadApiKey();
  }, []);

  // 当模态框打开时初始化状态
  useEffect(() => {
    if (open) {
      const currentConfig = { ...configRef.current };
      setOriginalConfig(currentConfig);
      setProviderType(currentConfig.provider_type || '');
      
      // 如果是 alibaba 类型，显示已保存的 API key
      if (currentConfig.provider_type === 'alibaba') {
        setApiKey(savedApiKey);
      } else {
        setApiKey('');
      }
    }
  }, [open, savedApiKey]);

  const renderEmpty = useCallback(() => {
    return (
      <div className="flex flex-col items-center">
        <Image
          alt="SecondMe Logo"
          className="object-contain"
          height={40}
          src="/images/single_logo.png"
          width={120}
        />
        <div className="text-gray-500 text-[18px] leading-[32px]">
          Please Choose Alibaba Cloud Model Studio
        </div>
      </div>
    );
  }, []);

  const renderAlibabaCloud = useCallback(() => {
    return (
      <div className="flex flex-col w-full gap-4">
        <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
            <Input.Password
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your Alibaba Cloud API Key"
              value={apiKey}
            />
            <div className="mt-2 text-sm text-gray-500">
              You can find your API key in your{' '}
              <a
                className="text-blue-500 hover:underline"
                href="https://bailian.console.aliyun.com/console?tab=model#/api-key"
                rel="noopener noreferrer"
                target="_blank"
              >
                Alibaba Cloud Model Studio API Keys page
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }, [apiKey]);

  const renderNoneProvider = useCallback(() => {
    return (
      <div className="flex flex-col items-center">
        <Image
          alt="SecondMe Logo"
          className="object-contain"
          height={40}
          src="/images/single_logo.png"
          width={120}
        />
        <div className="text-gray-500 text-[18px] leading-[32px]">
          None Cloud Provider Configured
        </div>
        <div className="text-sm text-gray-400 mt-2 text-center">
          No cloud provider will be used for training. You can select "Model Studio" to configure Alibaba Cloud.
        </div>
      </div>
    );
  }, []);

  const handleUpdate = async (): Promise<void> => {
    try {
      setLoading(true);

      if (providerType === 'alibaba' && apiKey.trim()) {
        const modelConfigResponse = await getModelConfig();

        if (modelConfigResponse.data?.data) {
          const currentConfig = modelConfigResponse.data.data;

          await updateModelConfig({
            ...currentConfig,
            cloud_service_api_key: apiKey.trim()
          });

          updateConfigRef.current({
            ...configRef.current,
            provider_type: 'alibaba',
            cloud_service_api_key: apiKey.trim()
          });

          setSavedApiKey(apiKey.trim());
          message.success('API key has been successfully saved');
        }
      } else if (providerType === '') {
        const modelConfigResponse = await getModelConfig();

        if (modelConfigResponse.data?.data) {
          const currentConfig = modelConfigResponse.data.data;

          await updateModelConfig({
            ...currentConfig,
            cloud_service_api_key: ''
          });

          updateConfigRef.current({
            ...configRef.current,
            provider_type: '',
            cloud_service_api_key: ''
          });

          setSavedApiKey('');
          message.success('Cloud provider configuration removed');
        }
      }

      await saveCloudConfig();
      onClose();
    } catch (error) {
      console.error('Error saving configuration:', error);
      message.error('An error occurred while saving configuration');
    } finally {
      setLoading(false);
    }
  };

  const renderMainContent = useCallback(() => {
    if (providerType === 'alibaba') {
      return renderAlibabaCloud();
    } else if (providerType === '') {
      return renderNoneProvider();
    }
    return renderEmpty();
  }, [providerType, renderAlibabaCloud, renderNoneProvider, renderEmpty]);

  const handleCancel = useCallback(() => {
    if (originalConfig) {
      updateConfigRef.current(originalConfig);
      setProviderType(originalConfig.provider_type || '');
      setApiKey(originalConfig.cloud_service_api_key || '');
    }
    onClose();
  }, [onClose, originalConfig]);

  const handleProviderTypeChange = useCallback((e: any) => {
    const newProviderType = e.target.value;
    setProviderType(newProviderType);

    // 根据选择的提供商类型设置 API key
    if (newProviderType === 'alibaba') {
      setApiKey(savedApiKey); // 恢复已保存的 API key
    } else {
      setApiKey(''); // 清空 API key
    }
  }, [savedApiKey]);

  // 计算 OK 按钮是否应该被禁用
  const isOkDisabled = 
    loading || 
    providerType === '' || // 选择了 "None" 时禁用
    (providerType === 'alibaba' && !apiKey.trim()); // 选择了 "Model Studio" 但没有 API key

  return (
    <Modal
      centered
      confirmLoading={loading}
      destroyOnClose
      okButtonProps={{
        disabled: isOkDisabled
      }}
      onCancel={handleCancel}
      onOk={handleUpdate}
      open={open}
      title={
        <div className="flex items-center gap-2">
          <div className="text-xl font-semibold leading-6 text-gray-900">
            Cloud Provider Configuration
          </div>
          <a
            className="text-gray-500 hover:text-gray-700"
            href="https://secondme.gitbook.io/secondme/guides/create-second-me/cloud-provider-config"
            rel="noreferrer"
            target="_blank"
          >
            <QuestionCircleOutlined />
          </a>
        </div>
      }
    >
      <div className="flex flex-col items-center">
        <div className="flex flex-col items-center gap-2">
          <p className="mb-1 text-sm text-gray-500">
            Configure cloud provider for model training with Second Me. Training with cloud
            providers typically completes faster but may incur costs.
          </p>
          <Radio.Group
            buttonStyle="solid"
            onChange={handleProviderTypeChange}
            optionType="button"
            options={options}
            value={providerType}
          />
        </div>
        <div className="w-full border-t border-gray-200 mt-1 mb-2" />
        {loading ? (
          <div className="flex items-center justify-center h-40">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        ) : (
          renderMainContent()
        )}
        <div className="w-full border-t border-gray-200 mt-4" />
      </div>
    </Modal>
  );
};


export default CloudProviderModal;
