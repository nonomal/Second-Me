import { Status, statusRankMap, useTrainingStore } from '@/store/useTrainingStore';
import { startService, stopService } from '@/service/train';
import { StatusBar } from '../StatusBar';
import { useRef, useEffect, useState, useMemo } from 'react';
import { message, Select, Tooltip, Spin, Modal, Tabs, List, Typography, Empty } from 'antd';
import {
  CloudUploadOutlined,
  CheckCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  LoadingOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import RegisterUploadModal from '../upload/RegisterUploadModal';
import Image from 'next/image';

import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import TrainingTipModal from '../upload/TraingTipModal';
import { getMemoryList } from '@/service/memory';
import { getModelList, ModelInfo } from '@/service/model';

const { Text } = Typography;

// ModelSelectionModal Component
interface ModelSelectionModalProps {
  open: boolean;
  handleClose: () => void;
  onModelSelect: (model: ModelInfo) => void;
  localModels: ModelInfo[];
  loadingModels: boolean;
}

const ModelSelectionModal: React.FC<ModelSelectionModalProps> = ({
  open,
  handleClose,
  onModelSelect,
  localModels,
  loadingModels
}) => {
  // Mock cloud models - in production this would come from an API
  const [cloudModels, setCloudModels] = useState<ModelInfo[]>([]);
  const [loadingCloudModels, setLoadingCloudModels] = useState<boolean>(false);

  // Custom empty component
  const renderEmpty = (type: 'local' | 'cloud') => {
    const messageText = type === 'local' ? 'No local models available' : 'No cloud models available';

    return (
      <div className="flex flex-col items-center">
        <Image
          alt="SecondMe Logo"
          className="object-contain"
          height={40}
          src="/images/single_logo.png"
          width={120}
        />
        <div className="text-gray-500 text-[18px] leading-[32px]">{messageText}</div>
      </div>
    );
  };

  // For demonstration, we'll use a mock function to load cloud models
  useEffect(() => {
    if (open) {
      // Here you would fetch cloud models from your API
      setLoadingCloudModels(true);

      // Simulate API call with timeout
      const timeout = setTimeout(() => {
        // Mock cloud models for demonstration
        // In real implementation, this would be replaced with actual API call
        const mockCloudModels: ModelInfo[] = [
          {
            model_path: 'cloud/gpt-4-turbo-20240509',
            full_path: '/models/cloud/gpt-4-turbo-20240509',
            file_size: 0, // Not applicable for cloud models
            created_time: Date.now() - 7 * 24 * 60 * 60 * 1000, // 1 week ago
            training_params: {
              type: 'pretrained',
              tokens: '1.5T',
              architecture: 'Transformer',
              context_length: '128K'
            }
          },
          {
            model_path: 'cloud/llama-3-70b-instruct',
            full_path: '/models/cloud/llama-3-70b-instruct',
            file_size: 0, // Not applicable for cloud models
            created_time: Date.now() - 14 * 24 * 60 * 60 * 1000, // 2 weeks ago
            training_params: {
              type: 'fine-tuned',
              tokens: '8T',
              architecture: 'Transformer',
              context_length: '8K'
            }
          }
        ];

        setCloudModels(mockCloudModels);
        setLoadingCloudModels(false);
      }, 1000);

      return () => clearTimeout(timeout);
    }
  }, [open]);

  // Sort models to get the newest one first
  const sortedLocalModels = useMemo(() => {
    if (!localModels || localModels.length === 0) return [];

    return [...localModels].sort((a, b) => {
      const fileNameA = a.model_path.split('/')[1] || '';
      const fileNameB = b.model_path.split('/')[1] || '';

      // Assume newer models have higher timestamp or more recent name
      return fileNameB.localeCompare(fileNameA);
    });
  }, [localModels]);

  // Sort cloud models to get the newest one first
  const sortedCloudModels = useMemo(() => {
    if (!cloudModels || cloudModels.length === 0) return [];

    return [...cloudModels].sort((a, b) => {
      // Sort by created_time (newest first)
      return b.created_time - a.created_time;
    });
  }, [cloudModels]);

  // Get the newest local model (first in sorted array)
  const newestLocalModel = sortedLocalModels.length > 0 ? sortedLocalModels[0] : null;

  // Get the newest cloud model (first in sorted array)
  const newestCloudModel = sortedCloudModels.length > 0 ? sortedCloudModels[0] : null;

  const renderModelItem = (model: ModelInfo, index: number, isCloud: boolean = false) => {
    const fileName = model.model_path.split('/')[1];
    const timeStamp = fileName?.replace('.gguf', '') || 'Unknown version';
    const modelName = model.model_path.split('/')[0];

    // Check if this is the newest model in its category
    const isNewestLocal =
      !isCloud && newestLocalModel && model.model_path === newestLocalModel.model_path;
    const isNewestCloud =
      isCloud && newestCloudModel && model.model_path === newestCloudModel.model_path;
    const showNewTag = isNewestLocal || isNewestCloud;

    return (
      <div
        key={model.model_path}
        className="relative cursor-pointer hover:bg-gray-50 rounded transition-colors p-3 border border-gray-200 mb-3"
        onClick={() => onModelSelect(model)}
      >
        <div className="flex items-center justify-between w-full">
          <div className="flex flex-col">
            <Text strong>{timeStamp}</Text>
            <Text className="text-xs" type="secondary">
              {modelName}
            </Text>
            {isCloud && (
              <Text className="text-xs text-purple-500 mt-1" type="secondary">
                Cloud Model
              </Text>
            )}
          </div>

          <div className="flex items-center space-x-2">
            {showNewTag && (
              <div className="px-2 py-0.5 bg-blue-500 text-white text-xs rounded-full font-medium shadow-sm">
                New
              </div>
            )}
            <Tooltip
              placement="left"
              title={
                <div className="text-xs">
                  <div className="font-bold mb-1">{modelName}</div>
                  <div className="font-bold mt-2 mb-1">Training Parameters:</div>
                  <ul>
                    {Object.entries(model.training_params).length > 0 ? (
                      Object.entries(model.training_params).map(([key, value]) => (
                        <li key={key}>
                          {key}: {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </li>
                      ))
                    ) : (
                      <li>No training parameters available</li>
                    )}
                  </ul>
                </div>
              }
            >
              <InfoCircleOutlined className="text-gray-400 hover:text-blue-500" />
            </Tooltip>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Modal
      bodyStyle={{ maxHeight: '70vh', overflow: 'hidden', padding: '20px' }}
      footer={null}
      onCancel={handleClose}
      open={open}
      title="Select a Model to Start"
      width={600}
    >
      <div className="overflow-hidden">
        <div className="overflow-auto no-scrollbar" style={{ maxHeight: 'calc(70vh - 120px)' }}>
          {/* Local Models Section */}
          <div className="mb-6">
            <div className="text-base font-medium mb-2">Local Models</div>
            <Spin spinning={loadingModels}>
              {sortedLocalModels.length > 0 ? (
                <div>{sortedLocalModels.map((model, index) => renderModelItem(model, index))}</div>
              ) : (
                renderEmpty('local')
              )}
            </Spin>
          </div>

          {/* Cloud Models Section */}
          <div>
            <div className="text-base font-medium mb-2">Cloud Models</div>
            <div className="text-sm text-gray-500 mb-4">
              All cloud models feature high-efficiency training and are billed using a token-based
              pricing model.
            </div>
            <Spin spinning={loadingCloudModels}>
              {sortedCloudModels.length > 0 ? (
                <div>
                  {sortedCloudModels.map((model, index) => renderModelItem(model, index, true))}
                </div>
              ) : (
                renderEmpty('cloud')
              )}
            </Spin>
          </div>
        </div>
      </div>

      <style global jsx>{`
        .no-scrollbar {
          -ms-overflow-style: none;  /* IE and Edge */
          scrollbar-width: none;  /* Firefox */
        }
        .no-scrollbar::-webkit-scrollbar {
          display: none;  /* Chrome, Safari and Opera */
        }
      `}</style>
    </Modal>
  );
};

const StatusDot = ({ active }: { active: boolean }) => (
  <div
    className={`w-2 h-2 rounded-full mr-2 transition-colors duration-300 ${active ? 'bg-[#52c41a]' : 'bg-[#ff4d4f]'}`}
  />
);

// Helper component for option tooltips

export function ModelStatus() {
  const status = useTrainingStore((state) => state.status);
  const setStatus = useTrainingStore((state) => state.setStatus);
  const serviceStarted = useTrainingStore((state) => state.serviceStarted);
  const isServiceStarting = useTrainingStore((state) => state.isServiceStarting);
  const isServiceStopping = useTrainingStore((state) => state.isServiceStopping);
  const setServiceStarting = useTrainingStore((state) => state.setServiceStarting);
  const setServiceStopping = useTrainingStore((state) => state.setServiceStopping);
  const fetchServiceStatus = useTrainingStore((state) => state.fetchServiceStatus);
  const isTraining = useTrainingStore((state) => state.isTraining);

  const [messageApi, contextHolder] = message.useMessage();

  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);

  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [showtrainingModal, setShowtrainingModal] = useState(false);
  const [showModelSelectionModal, setShowModelSelectionModal] = useState(false);
  const [showConfirmationModal, setShowConfirmationModal] = useState(false);
  const [selectedModelForConfirmation, setSelectedModelForConfirmation] = useState<ModelInfo | null>(null);

  // Model selection states
  const [modelList, setModelList] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loadingModels, setLoadingModels] = useState<boolean>(false);

  const handleRegistryClick = () => {
    if (!serviceStarted) {
      messageApi.info({
        content: 'Please start your model service first',
        duration: 1
      });
    } else {
      setShowRegisterModal(true);
    }
  };


  const fetchMemories = async () => {
    try {
      const memoryRes = await getMemoryList();

      if (memoryRes.data.code === 0) {
        const memories = memoryRes.data.data;

        if (memories.length > 0 && statusRankMap[status] < statusRankMap[Status.MEMORY_UPLOAD]) {
          setStatus(Status.MEMORY_UPLOAD);
        }
      }
    } catch (error: any) {
      console.error('Error fetching memories:', error);
    }
  };

  const fetchModels = async () => {
    setLoadingModels(true);

    try {
      const res = await getModelList();

      if (res.data.code === 0) {
        setModelList(res.data.data);
        // Read saved configuration, initialize selected model
        const config = JSON.parse(localStorage.getItem('trainingParams') || '{}');

        if (config.model_name) {
          // Check if the model exists in the list
          const modelExists = res.data.data.some(
            (model: ModelInfo) => model.model_path === config.model_name
          );

          if (modelExists) {
            setSelectedModel(config.model_name);
          } else if (res.data.data.length > 0) {
            // Default to the first model in the list
            setSelectedModel(res.data.data[0].model_path);
          }
        } else if (res.data.data.length > 0) {
          // Default to the first model in the list
          setSelectedModel(res.data.data[0].model_path);
        }
      } else {
        messageApi.error({ content: res.data.message!, duration: 1 });
      }
    } catch (error: unknown) {
      console.error('Error fetching model list:', error);
      messageApi.error({
        content: error.response?.data?.message || error.message,
        duration: 1
      });
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    fetchMemories();
    fetchServiceStatus();
    fetchModels();

    return () => {
      clearPolling();
    };
  }, []);

  const pollingInterval = useRef<NodeJS.Timeout | null>(null);

  const clearPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  };

  const startPolling = () => {
    clearPolling();

    // Start new polling interval
    pollingInterval.current = setInterval(() => {
      fetchServiceStatus()
        .then((res: any) => {
          if (res && res.data && res.data.code === 0) {
            const isRunning = res.data.data.is_running;

            if (isRunning) {
              setServiceStarting(false);
              clearPolling();
            }
          }
        })
        .catch((error: any) => {
          console.error('Error checking service status:', error);
        });
    }, 3000);
  };

  const startStopPolling = () => {
    clearPolling();

    // Start new polling interval
    pollingInterval.current = setInterval(() => {
      fetchServiceStatus()
        .then((res) => {
          if (res.data.code === 0) {
            const isRunning = res.data.data.is_running;

            if (!isRunning) {
              setServiceStopping(false);
              clearPolling();
            }
          }
        })
        .catch((error: any) => {
          console.error('Error checking service status:', error);
        });
    }, 3000);
  };

  const handleStopService = () => {
    setServiceStopping(true);
    stopService()
      .then((res) => {
        if (res.data.code === 0) {
          messageApi.success({ content: 'Service stopping...', duration: 1 });
          startStopPolling();
        } else {
          messageApi.error({ content: res.data.message!, duration: 1 });
          setServiceStopping(false);
        }
      })
      .catch((error: any) => {
        console.error('Error stopping service:', error);
        messageApi.error({
          content: error.response?.data?.message || error.message,
          duration: 1
        });
        setServiceStopping(false);
      });
  };

  const handleServiceAction = () => {
    if (serviceStarted) {
      handleStopService();
    } else {
      if (isTraining) {
        setShowtrainingModal(true);
        return;
      }

      // Fetch latest models before showing model selection modal
      fetchModels().then(() => {
        // Show model selection modal after fetching the latest models
        setShowModelSelectionModal(true);
      });
    }
  };

  const handleModelSelect = (model: ModelInfo) => {
    setSelectedModel(model.model_path);

    // Store the complete model path in localStorage
    const config = JSON.parse(localStorage.getItem('trainingParams') || '{}');
    config.model_name = model.model_path;
    localStorage.setItem('trainingParams', JSON.stringify(config));

    // Do not close the model selection modal, just show confirmation on top
    // Set the selected model for confirmation and show confirmation modal
    setSelectedModelForConfirmation(model);
    setShowConfirmationModal(true);
  };

  const handleStartServiceConfirmed = () => {
    if (!selectedModelForConfirmation) return;

    // Start the service with the selected model
    setServiceStarting(true);
    startService(selectedModelForConfirmation)
      .then((res) => {
        if (res.data.code === 0) {
          messageApi.success({ content: 'Service starting...', duration: 1 });
          startPolling();
        } else {
          setServiceStarting(false);
          messageApi.error({ content: res.data.message!, duration: 1 });
        }
      })
      .catch((error: any) => {
        console.error('Error starting service:', error);
        setServiceStarting(false);
        messageApi.error({
          content: error.response?.data?.message || error.message,
          duration: 1
        });
      });

    // Close the confirmation modal
    setShowConfirmationModal(false);
    // Also close the model selection modal after successful confirmation
    setShowModelSelectionModal(false);
    setSelectedModelForConfirmation(null);
  };

  return (
    <div className="flex items-center justify-center gap-4 mx-auto">
      {contextHolder}
      <StatusBar status={status} />

      <div className="flex items-center gap-6">
        {/* Control Buttons */}
        <div className="flex items-center gap-3">
          <div
            className={`
              flex items-center space-x-1.5 text-sm whitespace-nowrap
              ${
                isServiceStarting || isServiceStopping
                  ? 'text-gray-400 cursor-not-allowed'
                  : 'text-gray-600 hover:text-blue-600 cursor-pointer transition-all hover:-translate-y-0.5'
              }
            `}
            onClick={isServiceStarting || isServiceStopping ? undefined : handleServiceAction}
          >
            {isServiceStarting || isServiceStopping ? (
              <>
                <LoadingOutlined className="text-lg" spin />
                <span>{isServiceStarting ? 'Starting...' : 'Stopping...'}</span>
              </>
            ) : serviceStarted ? (
              <>
                <StatusDot active={true} />
                <PauseCircleOutlined className="text-lg" />
                <span>Stop Service</span>
              </>
            ) : (
              <>
                <StatusDot active={false} />
                <PlayCircleOutlined className="text-lg" />
                <span>Start Service</span>
              </>
            )}
          </div>

          <div className="w-px h-4 bg-gray-200" />

          <div
            className="flex items-center whitespace-nowrap space-x-1.5 text-sm text-gray-600 hover:text-blue-600 cursor-pointer transition-all hover:-translate-y-0.5 mr-2"
            onClick={handleRegistryClick}
          >
            {isRegistered ? (
              <>
                <StatusDot active={true} />
                <CheckCircleOutlined className="text-lg" />
                <span>Join AI Network</span>
              </>
            ) : (
              <>
                <StatusDot active={false} />
                <CloudUploadOutlined className="text-lg" />
                <span>Join AI Network</span>
              </>
            )}
          </div>
        </div>
      </div>

      <RegisterUploadModal onClose={() => setShowRegisterModal(false)} open={showRegisterModal} />
      <TrainingTipModal
        confirm={() => {
          setShowModelSelectionModal(true);
          setShowtrainingModal(false);
        }}
        onClose={() => setShowtrainingModal(false)}
        open={showtrainingModal}
      />
      <ModelSelectionModal
        handleClose={() => setShowModelSelectionModal(false)}
        loadingModels={loadingModels}
        localModels={modelList}
        onModelSelect={handleModelSelect}
        open={showModelSelectionModal}
      />

      {/* Confirmation Modal for Model Selection */}
      <Modal
        cancelText="Cancel"
        centered
        okButtonProps={{
          style: { backgroundColor: '#1890ff', borderColor: '#1890ff' }
        }}
        okText="Start Service"
        onCancel={() => setShowConfirmationModal(false)}
        onOk={handleStartServiceConfirmed}
        open={showConfirmationModal}
        title={<div className="text-lg font-medium">Confirm Service Start</div>}
        width={480}
      >
        <div className="py-4">
          {selectedModelForConfirmation && (
            <>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className="bg-blue-50 p-3 rounded-lg mr-4">
                    {selectedModelForConfirmation.model_path.includes('cloud/') ? (
                      <svg
                        fill="none"
                        height="24"
                        stroke="#1890ff"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        viewBox="0 0 24 24"
                        width="24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" />
                      </svg>
                    ) : (
                      <svg
                        fill="none"
                        height="24"
                        stroke="#1890ff"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        viewBox="0 0 24 24"
                        width="24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <rect height="16" rx="2" ry="2" width="16" x="4" y="4" />
                        <rect height="6" width="6" x="9" y="9" />
                        <line x1="9" x2="9" y1="2" y2="4" />
                        <line x1="15" x2="15" y1="2" y2="4" />
                        <line x1="9" x2="9" y1="20" y2="22" />
                        <line x1="15" x2="15" y1="20" y2="22" />
                        <line x1="20" x2="22" y1="9" y2="9" />
                        <line x1="20" x2="22" y1="14" y2="14" />
                        <line x1="2" x2="4" y1="9" y2="9" />
                        <line x1="2" x2="4" y1="14" y2="14" />
                      </svg>
                    )}
                  </div>
                  <div>
                    <Text strong className="text-base">
                      {selectedModelForConfirmation.model_path.split('/')[1]}
                    </Text>
                    <div className="flex items-center">
                      <Text type="secondary" className="text-sm">
                        {selectedModelForConfirmation.model_path.split('/')[0]}
                      </Text>
                      {selectedModelForConfirmation.model_path.includes('cloud/') && (
                        <span className="ml-2 px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full text-xs">
                          Cloud Model
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <Tooltip
                  placement="left"
                  title={
                    <div className="text-xs">
                      <div className="font-bold mb-1">
                        {selectedModelForConfirmation.model_path.split('/')[0]}
                      </div>
                      <div className="font-bold mt-2 mb-1">Training Parameters:</div>
                      <ul>
                        {Object.entries(selectedModelForConfirmation.training_params).length > 0 ? (
                          Object.entries(selectedModelForConfirmation.training_params).map(([key, value]) => (
                            <li key={key}>
                                {key}:{' '}
                                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </li>
                          ))
                        ) : (
                          <li>No training parameters available</li>
                        )}
                      </ul>
                    </div>
                  }
                >
                  <InfoCircleOutlined className="text-gray-400 hover:text-blue-500 text-xl cursor-pointer" />
                </Tooltip>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg text-sm">
                {selectedModelForConfirmation.model_path.includes('cloud/') ? (
                  <div>
                    <p className="mb-2 font-medium text-gray-700">
                      Starting{' '}
                      <span className="font-bold text-blue-600">
                        {selectedModelForConfirmation.model_path.split('/')[1]}
                      </span>{' '}
                      will incur charges on your Alibaba Cloud Model Studio account and will be
                      billed continuously based on token usage.
                    </p>
                    <p className="text-gray-600">Do you want to continue?</p>
                  </div>
                ) : (
                  <div>
                    <p className="mb-2 font-medium text-gray-700">
                      You are about to start the local model{' '}
                      <span className="font-bold text-blue-600">
                        {selectedModelForConfirmation.model_path.split('/')[1]}
                      </span>
                      . This will run the model on your local machine using the available resources.
                    </p>
                    <p className="text-gray-600">Do you want to continue?</p>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </Modal>
    </div>
  );
}
