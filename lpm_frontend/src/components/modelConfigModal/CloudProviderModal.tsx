import { Modal, Radio, Input, message } from 'antd';
import Image from 'next/image';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useCallback, useEffect, useState, useRef } from 'react';
import { setCloudServiceApiKey, getCloudServiceApiKey } from '../../service/modelConfig';

interface CloudProviderConfig {
  provider_type: string;
  api_key?: string;
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
  const [loading, setLoading] = useState<boolean>(false);
  const [originalConfig, setOriginalConfig] = useState<CloudProviderConfig | null>(null);

  // 使用useRef存储最新的props，避免useEffect依赖过多
  const configRef = useRef(cloudConfig);
  const updateConfigRef = useRef(updateCloudConfig);

  // 更新ref的值
  useEffect(() => {
    configRef.current = cloudConfig;
    updateConfigRef.current = updateCloudConfig;
  }, [cloudConfig, updateCloudConfig]);

  // 只在modal打开时获取API密钥
  useEffect(() => {
    if (open) {
      // 保存原始配置
      setOriginalConfig({ ...configRef.current });
      setProviderType(configRef.current.provider_type || '');

      // 如果选择了阿里云，请求API密钥
      if (configRef.current.provider_type === 'alibaba') {
        setLoading(true);
        getCloudServiceApiKey()
          .then((res) => {
            if (res.data.data.api_key) {
              // 更新originalConfig
              setOriginalConfig({
                ...configRef.current,
                api_key: res.data.data.api_key
              });
              // 更新当前显示的值
              updateConfigRef.current({
                ...configRef.current,
                api_key: res.data.data.api_key
              });
            }
          })
          .catch((error) => {
            console.error('Failed to get API key:', error);
          })
          .finally(() => {
            setLoading(false);
          });
      }
    }
  }, [open]);

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
                updateConfigRef.current({ ...configRef.current, api_key: e.target.value });
              }}
              placeholder="Enter your Alibaba Cloud API Key"
              value={configRef.current.api_key}
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
  }, []);

  const handleUpdate = async (): Promise<void> => {
    try {
      setLoading(true);

      // 如果选择了阿里云，并且有API密钥，则调用设置API密钥的接口
      if (providerType === 'alibaba' && configRef.current.api_key) {
        await setCloudServiceApiKey(configRef.current.api_key);
        message.success('API key has been successfully saved');
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
    if (!providerType) {
      return renderEmpty();
    }

    if (providerType === 'alibaba') {
      return renderAlibabaCloud();
    }

    return renderEmpty();
  }, [providerType, renderAlibabaCloud]);

  const handleCancel = useCallback(() => {
    // 恢复原始配置
    if (originalConfig) {
      updateConfigRef.current(originalConfig);
    }
    onClose();
  }, [onClose, originalConfig]);

  return (
    <Modal
      centered
      destroyOnClose
      okButtonProps={{ disabled: !providerType || loading }}
      onCancel={handleCancel}
      onOk={handleUpdate}
      open={open}
      confirmLoading={loading}
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
              updateConfigRef.current({ 
                ...configRef.current, 
                provider_type: newProviderType,
                // 如果不是阿里云，清空API密钥
                api_key: newProviderType !== 'alibaba' ? '' : configRef.current.api_key
              });
            }}
            optionType="button"
            options={options}
            value={providerType}
          />
        </div>
        <div className="w-full border-t border-gray-200 mt-1 mb-2" />
        {renderMainContent()}
        <div className="w-full border-t border-gray-200 mt-4" />
      </div>
    </Modal>
  );
};

export default CloudProviderModal;
