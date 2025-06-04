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
  const [initialLoadDone, setInitialLoadDone] = useState<boolean>(false);

  const configRef = useRef(cloudConfig);
  const updateConfigRef = useRef(updateCloudConfig);

  useEffect(() => {
    configRef.current = cloudConfig;
    updateConfigRef.current = updateCloudConfig;
  }, [cloudConfig, updateCloudConfig]);

  // 立即获取最新配置，不等待模态框打开
  useEffect(() => {
    // 预加载API key，提前获取
    const preloadApiKey = async () => {
      try {
        const res = await getModelConfig();
        if (res.data.data && res.data.data.cloud_service_api_key) {
          // 只更新本地状态，不更新全局配置
          setApiKey(res.data.data.cloud_service_api_key);
          setInitialLoadDone(true);
        }
      } catch (error) {
        console.error('Failed to preload API key:', error);
      }
    };

    preloadApiKey();
  }, []);

  useEffect(() => {
    if (open) {
      setOriginalConfig({ ...configRef.current });
      setProviderType(configRef.current.provider_type || '');
      
      // 如果预加载已完成，使用预加载的apiKey
      if (initialLoadDone) {
        // 已经有预加载的key，不需要再次加载
        return;
      }

      if (configRef.current.provider_type === 'alibaba') {
        setLoading(true);
        getModelConfig()
          .then((res) => {
            if (res.data.data && res.data.data.cloud_service_api_key) {
              setOriginalConfig({
                ...configRef.current,
                cloud_service_api_key: res.data.data.cloud_service_api_key
              });
              setApiKey(res.data.data.cloud_service_api_key); // Set local API Key
            }
          })
          .catch((error) => {
            console.error('Failed to get API key:', error);
          })
          .finally(() => {
            setLoading(false);
            setInitialLoadDone(true);
          });
      }
    }
  }, [open, initialLoadDone]);

  const renderEmpty = () => {
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
  };

  const renderAlibabaCloud = useCallback(() => {
    return (
      <div className="flex flex-col w-full gap-4">
        <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
            <Input.Password
              onChange={(e) => {
                setApiKey(e.target.value); // Only update local state
              }}
              placeholder="Enter your Alibaba Cloud API Key"
              value={apiKey} // Use local state
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
  }, [apiKey]); // Add apiKey dependency

  const handleUpdate = async (): Promise<void> => {
    try {
      setLoading(true);

      if (providerType === 'alibaba' && apiKey) {
        const modelConfigResponse = await getModelConfig();

        if (modelConfigResponse.data.data) {
          const currentConfig = modelConfigResponse.data.data;

          await updateModelConfig({
            ...currentConfig,
            cloud_service_api_key: apiKey // Use local apiKey state
          });

          // Update global state to alibaba
          updateConfigRef.current({
            ...configRef.current,
            provider_type: 'alibaba',
            cloud_service_api_key: apiKey // Use local apiKey state
          });

          message.success('API key has been successfully saved');
        }
      } else if (providerType === '') {
        const modelConfigResponse = await getModelConfig();

        if (modelConfigResponse.data.data) {
          const currentConfig = modelConfigResponse.data.data;

          await updateModelConfig({
            ...currentConfig,
            cloud_service_api_key: ''
          });

          // Update global state to empty
          updateConfigRef.current({
            ...configRef.current,
            provider_type: '',
            cloud_service_api_key: ''
          });

          message.success('Cloud provider configuration removed');
        }
      }

      // Call save method and ensure state update
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
    if (!providerType) {
      return renderEmpty();
    }

    if (providerType === 'alibaba') {
      return renderAlibabaCloud();
    }

    return renderEmpty();
  }, [providerType, renderAlibabaCloud]);

  const handleCancel = useCallback(() => {
    if (originalConfig) {
      updateConfigRef.current(originalConfig);
      setProviderType(originalConfig.provider_type || '');
      setApiKey(originalConfig.cloud_service_api_key || ''); // Restore local API Key
    }
    onClose();
  }, [onClose, originalConfig]);

  return (
    <Modal
      centered
      confirmLoading={loading}
      destroyOnClose
      okButtonProps={{
        disabled: (providerType === 'alibaba' && !apiKey) || loading
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
            onChange={(e) => {
              const newProviderType = e.target.value;

              setProviderType(newProviderType);

              // If switching to non-alibaba type, clear local API Key
              if (newProviderType !== 'alibaba') {
                setApiKey('');
              }
            }}
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
