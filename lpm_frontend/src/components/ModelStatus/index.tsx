import { Status, statusRankMap, useTrainingStore } from '@/store/useTrainingStore';
import { startService, stopService } from '@/service/train';
import { StatusBar } from '../StatusBar';
import { useRef, useEffect, useState, useMemo } from 'react';
import { message, Select, Tooltip, Spin } from 'antd';
import {
  CloudUploadOutlined,
  CheckCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  LoadingOutlined
} from '@ant-design/icons';
import RegisterUploadModal from '../upload/RegisterUploadModal';

import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import TrainingTipModal from '../upload/TraingTipModal';
import { getMemoryList } from '@/service/memory';
import { getModelList, ModelInfo } from '@/service/model';

const StatusDot = ({ active }: { active: boolean }) => (
  <div
    className={`w-2 h-2 rounded-full mr-2 transition-colors duration-300 ${active ? 'bg-[#52c41a]' : 'bg-[#ff4d4f]'}`}
  />
);

// Helper component for option tooltips
const OptionTooltip = ({ model }: { model: ModelInfo }) => {
  const fileName = model.model_path.split('/')[1];
  const timeStamp = fileName?.replace('.gguf', '') || 'Unknown version';
  const modelName = model.model_path.split('/')[0];

  return (
    <Tooltip
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
      placement="left"
    >
      <div className="flex flex-col">
        <span className="font-medium">{timeStamp}</span>
        <span className="text-xs text-gray-500">{modelName}</span>
      </div>
    </Tooltip>
  );
};

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

  const handleModelChange = (value: string) => {
    setSelectedModel(value);
    // Update configuration in localStorage
    const config = JSON.parse(localStorage.getItem('trainingParams') || '{}');

    // Store the complete model path
    config.model_name = value;
    localStorage.setItem('trainingParams', JSON.stringify(config));
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
    } catch (error) {
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
    } catch (error) {
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
        .then((res) => {
          if (res.data.code === 0) {
            const isRunning = res.data.data.is_running;

            if (isRunning) {
              setServiceStarting(false);
              clearPolling();
            }
          }
        })
        .catch((error) => {
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
        .catch((error) => {
          console.error('Error checking service status:', error);
        });
    }, 3000);
  };  const handleStartService = () => {
    // Use selected model or get from local storage
    const config = JSON.parse(localStorage.getItem('trainingParams') || '{}');
    const modelPath = selectedModel || config.model_name;

    if (!modelPath) {
      message.error('Please select a model to start');
      return;
    }

    setServiceStarting(true);
    startService({ model_name: modelPath })
      .then((res) => {
        if (res.data.code === 0) {
          messageApi.success({ content: 'Service starting...', duration: 1 });
          startPolling();
        } else {
          setServiceStarting(false);
          messageApi.error({ content: res.data.message!, duration: 1 });
        }
      })
      .catch((error) => {
        console.error('Error starting service:', error);
        setServiceStarting(false);
        messageApi.error({
          content: error.response?.data?.message || error.message,
          duration: 1
        });
      });
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
      .catch((error) => {
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

      handleStartService();
    }
  };

  return (
    <div className="flex items-center justify-center gap-4 mx-auto">
      {contextHolder}
      <StatusBar status={status} />

      <div className="flex items-center gap-6">
        {/* Model Selection - only show when service is not started */}
        {!serviceStarted && (
          <div className="relative mr-3">
            <Spin spinning={loadingModels} size="small">
              <Select
                className="w-56"
                disabled={serviceStarted || isServiceStarting}
                loading={loadingModels}
                onChange={handleModelChange}
                optionLabelProp="label"
                placeholder="Select Model"
                value={selectedModel}
                options={modelList.map((model) => {
                  const fileName = model.model_path.split('/')[1];
                  const timeStamp = fileName?.replace('.gguf', '') || 'Unknown version';
                  const modelName = model.model_path.split('/')[0];
                  return {
                    label: timeStamp,
                    value: model.model_path,
                    model: model
                  };
                })}
                optionRender={(option) => {
                  const model = option.data.model as ModelInfo;
                  return <OptionTooltip model={model} />;
                }}
                onDropdownVisibleChange={(open) => {
                  if (open) {
                    // Hide any currently visible tooltips when dropdown opens
                    const tooltipElements = document.querySelectorAll('.ant-tooltip');
                    tooltipElements.forEach(el => {
                      if (window.getComputedStyle(el).display !== 'none') {
                        (el as HTMLElement).style.display = 'none';
                      }
                    });
                  }
                }}
              />
            </Spin>
          </div>
        )}
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
          handleStartService();
          setShowtrainingModal(false);
        }}
        onClose={() => setShowtrainingModal(false)}
        open={showtrainingModal}
      />
    </div>
  );
}
