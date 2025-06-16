import { Status, statusRankMap, useTrainingStore } from '@/store/useTrainingStore';
import { startService, stopService } from '@/service/train';
import { StatusBar } from '../StatusBar';
import { useRef, useEffect, useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
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
import {
  listDeployments,
  CloudDeployment,
  startCloudService,
  stopCloudService,
  getCloudServiceStatus
} from '@/service/cloudService';

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
      <div className="flex flex-col items-center py-8 px-4">
        <Image
          alt="SecondMe Logo"
          className="object-contain opacity-30 mb-2"
          height={32}
          src="/images/single_logo.png"
          width={96}
        />
        <div className="text-gray-500 text-base text-center">{messageText}</div>
        {type === 'local' && (
          <div className="text-gray-400 text-sm text-center mt-2">
            You can add local models to get started
          </div>
        )}
        {type === 'cloud' && (
          <div className="text-gray-400 text-sm text-center mt-2">
            No deployed cloud models found
          </div>
        )}
      </div>
    );
  };

  // Fetch cloud models from real API
  useEffect(() => {
    if (open) {
      setLoadingCloudModels(true);

      listDeployments()
        .then((res) => {
          if (res.data.code === 0 && res.data.data.deployments) {
            // Show all deployments, including both deployed and undeployed ones
            const deploymentModels: ModelInfo[] = res.data.data.deployments.map((deployment: CloudDeployment) => {
                // Extract timestamp from job_id
                const timestampMatch = deployment.job_id.match(/ft-(\d{12})/);
                let extractedTimestamp;
                if (timestampMatch) {
                  const rawTimestamp = timestampMatch[1];
                  const year = parseInt(rawTimestamp.slice(0, 4));
                  const month = parseInt(rawTimestamp.slice(4, 6)) - 1; 
                  const day = parseInt(rawTimestamp.slice(6, 8));
                  const hour = parseInt(rawTimestamp.slice(8, 10));
                  const minute = parseInt(rawTimestamp.slice(10, 12));
                  extractedTimestamp = new Date(year, month, day, hour, minute).getTime() / 1000;
                } else {
                  extractedTimestamp = Date.now() / 1000;
                }
                
                const modelInfo = {
                  model_path: `cloud/${deployment.job_id}`, 
                  full_path: `/models/cloud/${deployment.job_id}`,
                  file_size: 0, 
                  created_time: extractedTimestamp,
                  training_params: {
                    type: 'fine-tuned',
                    base_model: deployment.base_model,
                    job_id: deployment.job_id,
                    deployed_model: deployment.deployed_model,
                    name: deployment.name,
                    training_type: deployment.training_type,
                    is_deployed: deployment.is_deployed,
                    usage: deployment.usage,
                    hyper_parameters: deployment.hyper_parameters
                  }
                };
                
                return modelInfo;
            });
            
            setCloudModels(deploymentModels);
          } else {
            setCloudModels([]);
          }
        })
        .catch((error) => {
          console.error('Failed to fetch cloud deployments:', error);
          setCloudModels([]);
        })
        .finally(() => {
          setLoadingCloudModels(false);
        });
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
    const isCloudDeployment = isCloud || (fileName && model.model_path.startsWith('cloud/'));
    const timeStamp = fileName?.replace('.gguf', '') || 'Unknown version';
    const modelName = model.model_path.split('/')[0];
    
    // Check deployment status for cloud models
    const isDeployed = isCloudDeployment ? (model.training_params.is_deployed && model.training_params.deployed_model) : true;

    // Check if this is the newest model in its category
    const isNewestLocal =
      !isCloud && newestLocalModel && model.model_path === newestLocalModel.model_path;
    const isNewestCloud =
      isCloud && newestCloudModel && model.model_path === newestCloudModel.model_path;
    const showNewTag = isNewestLocal || isNewestCloud;

    return (
      <div
        key={model.model_path}
        className={`relative cursor-pointer hover:bg-white/80 rounded-lg transition-all duration-200 p-3 border mb-2 hover:shadow-md hover:border-blue-300 ${
          isCloud 
            ? isDeployed 
              ? 'border-blue-200 bg-blue-50/50 hover:bg-blue-50/80' 
              : 'border-orange-200 bg-orange-50/50 hover:bg-orange-50/80'
            : 'border-gray-200 bg-white/60 hover:bg-white/90'
        }`}
        onClick={() => onModelSelect(model)}
      >
        <div className="flex items-center justify-between w-full">
          <div className="flex flex-col flex-1">
            <div className="flex items-center mb-1">
              <Text strong className="text-sm">{timeStamp}</Text>
              {showNewTag && (
                <div className="ml-2 px-2 py-0.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white text-xs rounded-full font-medium shadow-sm">
                  New
                </div>
              )}
              {isCloudDeployment && (
                <div className={`ml-2 px-2 py-0.5 text-xs rounded-full font-medium ${
                  isDeployed 
                    ? 'bg-green-100 text-green-700 border border-green-200' 
                    : 'bg-orange-100 text-orange-700 border border-orange-200'
                }`}>
                  {isDeployed ? 'Deployed' : 'Not Deployed'}
                </div>
              )}
            </div>
            <Text className="text-xs text-gray-600" type="secondary">
              {isCloudDeployment && model.training_params.base_model ? model.training_params.base_model : modelName}
            </Text>
            {isCloudDeployment && !isDeployed && (
              <Text className="text-xs text-orange-600 mt-1">
                Please deploy this model in DashScope Console first
              </Text>
            )}
          </div>

          <div className="flex items-center space-x-2 ml-4">
            <Tooltip
              placement="left"
              title={
                <div className="text-xs max-w-xs">
                  <div className="font-semibold text-white mb-2 text-sm border-b border-white/20 pb-2">
                    {timeStamp}
                  </div>
                  
                  {isCloudDeployment && (
                    <div className="mb-2">
                      <div className="text-blue-200 text-xs mb-1">Deployment Status</div>
                      <div className={`text-xs ${isDeployed ? 'text-green-300' : 'text-orange-300'}`}>
                        {isDeployed ? 'Ready to use' : 'Requires deployment in DashScope Console'}
                      </div>
                    </div>
                  )}
                  
                  {model.training_params.base_model && (
                    <div className="mb-2">
                      <div className="text-blue-200 text-xs mb-1">Base Model</div>
                      <div className="text-white/90 text-xs">{model.training_params.base_model}</div>
                    </div>
                  )}
                  
                  <div className="mb-2">
                    <div className="text-blue-200 text-xs mb-1">Created</div>
                    {(() => {
                      const formatTime = (timestamp: number) => {
                        const date = new Date(timestamp * 1000);
                        const year = date.getFullYear();
                        const month = String(date.getMonth() + 1).padStart(2, '0');
                        const day = String(date.getDate()).padStart(2, '0');
                        const hour = String(date.getHours()).padStart(2, '0');
                        const minute = String(date.getMinutes()).padStart(2, '0');
                        return `${year}-${month}-${day} ${hour}:${minute}`;
                      };

                      return <div className="text-white/90 text-xs">{formatTime(model.created_time)}</div>;
                    })()}
                  </div>

                  <div className="space-y-1 border-t border-white/10 pt-2">
                    {Object.entries(model.training_params).length > 0 ? (
                      (() => {
                        const priorityOrder = [
                          'type',
                          'training_type',
                          'is_deployed',
                          'deployed_model',
                          'name',
                          'usage'
                        ];

                        return Object.entries(model.training_params)
                          .filter(([key]) => {
                            if (key === 'base_model') return false; 
                            if (key === 'hyper_parameters') return false; // Too complex to display
                            return true;
                          })
                          .sort((a, b) => {
                            const indexA = priorityOrder.indexOf(a[0]);
                            const indexB = priorityOrder.indexOf(b[0]);

                            if (indexA !== -1 && indexB !== -1) return indexA - indexB;
                            if (indexA !== -1) return -1;
                            if (indexB !== -1) return 1;
                            return a[0].localeCompare(b[0]);
                          })
                          .map(([key, value]) => (
                            <div key={key} className="flex justify-between items-start">
                              <span className="text-blue-200 text-xs capitalize shrink-0">
                                {key.replace(/_/g, ' ')}:
                              </span>
                              <span className="text-white/90 text-xs ml-2 text-right break-all">
                                {key === 'is_deployed' 
                                  ? (value ? 'Yes' : 'No')
                                  : key === 'deployed_model'
                                  ? (value || 'Not deployed')
                                  : key === 'name'
                                  ? (value || 'No name assigned')
                                  : (typeof value === 'object' ? JSON.stringify(value) : String(value))
                                }
                              </span>
                            </div>
                          ));
                      })()
                    ) : (
                      <div className="text-white/70 text-xs">No parameters available</div>
                    )}
                  </div>
                </div>
              }
            >
              <InfoCircleOutlined className="text-gray-400 hover:text-blue-500 text-base transition-colors" />
            </Tooltip>
            <div className="text-gray-300">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
              </svg>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Scrollable container component
  const ScrollableModelList = ({ 
    models,
    loading,
    emptyType,
    isCloud = false,
    height = '180px'
  }: {
    models: ModelInfo[];
    loading: boolean;
    emptyType: 'local' | 'cloud';
    isCloud?: boolean;
    height?: string;
  }) => (
    <div 
      className={`border border-gray-200 rounded-lg ${isCloud ? 'bg-blue-50/20' : 'bg-gray-50/30'}`}
      style={{ height, minHeight: '150px' }}
    >
      {loading ? (
        <div className="flex items-center justify-center h-full">
          <Spin />
        </div>
      ) : (
        <>
          {models.length > 0 ? (
            <div
              className="subtle-scrollbar"
              style={{
                height: '100%',
                overflowY: 'scroll',
                padding: '12px'
              }}
            >
              {models.map((model, index) => renderModelItem(model, index, isCloud))}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full">{renderEmpty(emptyType)}</div>
          )}
        </>
      )}
    </div>
  );

  return (
    <Modal
      footer={null}
      onCancel={handleClose}
      open={open}
      title={<div className="text-xl font-bold text-gray-900">Select a Model to Start</div>}
      width={650}
      style={{ top: 20 }}
    >
      <div style={{ height: '60vh', maxHeight: '600px', minHeight: '550px' }}>
        {/* Local Models Section */}
        <div className="mb-6">
          <div className="text-base font-semibold mb-3 flex items-center text-gray-700 ">
            Local Models
            <span className="ml-3 text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full font-medium">
              {sortedLocalModels.length}
            </span>
          </div>
          <ScrollableModelList
            models={sortedLocalModels}
            loading={loadingModels}
            emptyType="local"
            height="200px"
          />
        </div>

        {/* Cloud Models Section */}
        <div>
          <div className="text-base font-semibold mb-2 flex items-center text-gray-700 ">
            Cloud Models
            <span className="ml-3 text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full font-medium">
              {sortedCloudModels.length}
            </span>
          </div>
          <div className="text-sm text-gray-500 mb-3 mt-2">
            All cloud models feature high-efficiency training and are billed using a token-based pricing model.
          </div>
          <ScrollableModelList
            models={sortedCloudModels}
            loading={loadingCloudModels}
            emptyType="cloud"
            isCloud={true}
            height="220px"
          />
        </div>
      </div>

      <style global jsx>{`
        .subtle-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(203, 213, 225, 0.3) transparent;
        }

        .subtle-scrollbar::-webkit-scrollbar {
          width: 4px;
        }

        .subtle-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }

        .subtle-scrollbar::-webkit-scrollbar-thumb {
          background-color: rgba(203, 213, 225, 0.4);
          border-radius: 2px;
          transition: background-color 0.2s ease;
        }

        .subtle-scrollbar::-webkit-scrollbar-thumb:hover {
          background-color: rgba(148, 163, 184, 0.6);
        }

        .subtle-scrollbar:hover::-webkit-scrollbar-thumb {
          background-color: rgba(203, 213, 225, 0.6);
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
  const router = useRouter();
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
  const [showDeploymentRequiredModal, setShowDeploymentRequiredModal] = useState(false);
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
        const processedModels = res.data.data.map((model: ModelInfo) => {
          const updatedTrainingParams = {
            ...model.training_params,
            base_model: model.training_params.base_model || model.training_params.model_name || 'Unknown'
          };
          
          return {
            ...model,
            training_params: updatedTrainingParams
          };
        });
        
        setModelList(processedModels);
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
    } catch (error: any) {
      console.error('Error fetching model list:', error);
      messageApi.error({
        content: error.response?.data?.message || error.message || 'Failed to fetch models',
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
          if (res.data) {
            const isRunning = res.data.is_running;

            if (!isRunning) {
              setServiceStopping(false);
              clearPolling();
              // Navigate to train page after service is stopped
              setTimeout(() => {
                router.push('/dashboard/train/training');
              }, 500);
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
    
    // Check if cloud service is running by calling the API
    getCloudServiceStatus()
      .then((cloudRes) => {
        const isCloudRunning = cloudRes.data.code === 0 && cloudRes.data.data.status === 'active';
        
        if (isCloudRunning) {
          // Stop cloud service
          return stopCloudService()
            .then((res) => {
              if (res.data.code === 0) {
                messageApi.success({ content: 'Cloud service stopped successfully!', duration: 2 });
                fetchServiceStatus();
                // Navigate to train page after stopping service
                setTimeout(() => {
                  router.push('/dashboard/train/training');
                }, 1000);
              } else {
                messageApi.error({ content: res.data.message || 'Failed to stop cloud service', duration: 2 });
              }
            })
            .catch((error: any) => {
              console.error('Error stopping cloud service:', error);
              messageApi.error({
                content: error.response?.data?.message || 'Failed to stop cloud service',
                duration: 2
              });
            });
        } else {
          // Stop local service
          return stopService()
            .then((res) => {
              if (res.data.code === 0) {
                messageApi.success({ content: 'Service stopping...', duration: 1 });
                startStopPolling();
              } else {
                messageApi.error({ content: res.data.message!, duration: 1 });
              }
            })
            .catch((error: any) => {
              console.error('Error stopping service:', error);
              messageApi.error({
                content: error.response?.data?.message || error.message,
                duration: 1
              });
            });
        }
      })
      .catch((error: any) => {
        console.error('Error checking service status:', error);
        // Fallback to local service stop
        return stopService()
          .then((res) => {
            if (res.data.code === 0) {
              messageApi.success({ content: 'Service stopping...', duration: 1 });
              startStopPolling();
            } else {
              messageApi.error({ content: res.data.message!, duration: 1 });
            }
          })
          .catch((error: any) => {
            console.error('Error stopping service:', error);
            messageApi.error({
              content: error.response?.data?.message || error.message,
              duration: 1
            });
          });
      })
      .finally(() => {
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

    // Check if it's a cloud model and if it's deployed
    const isCloudModel = model.model_path.includes('cloud/');
    const isDeployed = isCloudModel ? (model.training_params.is_deployed && model.training_params.deployed_model) : true;

    if (isCloudModel && !isDeployed) {
      // Show deployment required modal instead of confirmation
      setSelectedModelForConfirmation(model);
      setShowDeploymentRequiredModal(true);
      return;
    }

    // Do not close the model selection modal, just show confirmation on top
    // Set the selected model for confirmation and show confirmation modal
    setSelectedModelForConfirmation(model);
    setShowConfirmationModal(true);
  };

  const handleStartServiceConfirmed = () => {
    if (!selectedModelForConfirmation) return;

    const isCloudModel = selectedModelForConfirmation.model_path.includes('cloud/');

    if (isCloudModel) {
      setServiceStarting(true);

      // For deployed cloud models, we must use deployed_model
      const deployedModel = selectedModelForConfirmation.training_params.deployed_model;
      
      console.log('Cloud model deployment info:', {
        job_id: selectedModelForConfirmation.training_params.job_id,
        deployed_model: deployedModel,
        name: selectedModelForConfirmation.training_params.name,
        is_deployed: selectedModelForConfirmation.training_params.is_deployed
      });
      
      if (!deployedModel) {
        messageApi.error({ content: 'Model is not deployed. Please deploy it first in DashScope Console.', duration: 3 });
        setServiceStarting(false);
        return;
      }

      const cloudStartRequest = {
        deployment_model: deployedModel
      };
      
      console.log('Starting cloud service with request:', cloudStartRequest);

      startCloudService(cloudStartRequest)
        .then((res) => {
          if (res.data.code === 0) {
            messageApi.success({ content: 'Cloud service started successfully!', duration: 2 });
            fetchServiceStatus();
          } else {
            messageApi.error({ content: res.data.message || 'Failed to start cloud service', duration: 2 });
          }
        })
        .catch((error: any) => {
          console.error('Error starting cloud service:', error);
          messageApi.error({
            content: error.response?.data?.message || 'Failed to start cloud service',
            duration: 2
          });
        })
        .finally(() => {
          setServiceStarting(false);
        });
    } else {
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
    }

    // Close the confirmation modal
    setShowConfirmationModal(false);
    setShowModelSelectionModal(false);
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
                      {selectedModelForConfirmation.training_params.is_deployed && selectedModelForConfirmation.training_params.name 
                        ? selectedModelForConfirmation.training_params.name 
                        : selectedModelForConfirmation.training_params.job_id}
                    </Text>
                    <div className="flex items-center">
                      <Text type="secondary" className="text-sm">
                        {selectedModelForConfirmation.model_path.includes('cloud/') 
                          ? `Base Model: ${selectedModelForConfirmation.training_params.base_model}` 
                          : selectedModelForConfirmation.model_path.split('/')[0]}
                      </Text>
                      {selectedModelForConfirmation.model_path.includes('cloud/') && (
                        <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 border border-green-200 text-xs rounded-full font-medium">
                          Deployed
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <Tooltip
                  placement="left"
                  title={
                    <div className="text-xs max-w-xs">
                      <div className="font-semibold text-white mb-2 text-sm border-b border-white/20 pb-2">
                        {selectedModelForConfirmation.training_params.is_deployed && selectedModelForConfirmation.training_params.name 
                          ? selectedModelForConfirmation.training_params.name 
                          : selectedModelForConfirmation.training_params.job_id}
                      </div>
                      
                      {selectedModelForConfirmation.training_params.base_model && (
                        <div className="mb-2">
                          <div className="text-blue-200 text-xs mb-1">Base Model</div>
                          <div className="text-white/90 text-xs">{selectedModelForConfirmation.training_params.base_model}</div>
                        </div>
                      )}
                      
                      <div className="mb-2">
                        <div className="text-blue-200 text-xs mb-1">Created</div>
                        {(() => {
                          const formatTime = (timestamp: number) => {
                            const date = new Date(timestamp * 1000);
                            const year = date.getFullYear();
                            const month = String(date.getMonth() + 1).padStart(2, '0');
                            const day = String(date.getDate()).padStart(2, '0');
                            const hour = String(date.getHours()).padStart(2, '0');
                            const minute = String(date.getMinutes()).padStart(2, '0');
                            return `${year}-${month}-${day} ${hour}:${minute}`;
                          };

                          return <div className="text-white/90 text-xs">{formatTime(selectedModelForConfirmation.created_time)}</div>;
                        })()}
                      </div>
                      
                      <div className="space-y-1 border-t border-white/10 pt-2">
                        {Object.entries(selectedModelForConfirmation.training_params).length > 0 ? (
                          (() => {
                            const priorityOrder = [
                              'type',
                              'status',
                              'base_model',
                              'deployed_model'
                            ];
                            
                            return Object.entries(selectedModelForConfirmation.training_params)
                              .filter(([key]) => {
                                if (key === 'created_at') return false; 
                                if (key === 'base_model') return false; 
                                if (key === 'model_name') return false; 
                                if (!selectedModelForConfirmation.model_path.includes('cloud/') && key === 'model_path') return false;
                                return true;
                              })
                              .sort((a, b) => {
                                const indexA = priorityOrder.indexOf(a[0]);
                                const indexB = priorityOrder.indexOf(b[0]);
                                
                                if (indexA !== -1 && indexB !== -1) return indexA - indexB;
                                if (indexA !== -1) return -1;
                                if (indexB !== -1) return 1;
                                
                                return a[0].localeCompare(b[0]);
                              })
                              .map(([key, value]) => (
                                <div key={key} className="flex justify-between items-start">
                                  <span className="text-blue-200 text-xs capitalize shrink-0">
                                    {key.replace(/_/g, ' ')}:
                                  </span>
                                  <span className="text-white/90 text-xs ml-2 text-right break-all">
                                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                  </span>
                                </div>
                              ));
                          })()
                        ) : (
                          <div className="text-white/70 text-xs">No parameters available</div>
                        )}
                      </div>
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
                        {selectedModelForConfirmation.training_params.is_deployed && selectedModelForConfirmation.training_params.name 
                          ? selectedModelForConfirmation.training_params.name 
                          : selectedModelForConfirmation.training_params.job_id}
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

      {/* Deployment Required Modal */}
      <Modal
        cancelText="Close"
        centered
        footer={[
          <button
            key="close"
            className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
            onClick={() => setShowDeploymentRequiredModal(false)}
          >
            Close
          </button>,
          <button
            key="console"
            className="ml-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            onClick={() => {
              window.open('https://bailian.console.aliyun.com/?tab=model#/efm/model_deploy', '_blank');
              setShowDeploymentRequiredModal(false);
            }}
          >
            Go to DashScope Console
          </button>
        ]}
        onCancel={() => setShowDeploymentRequiredModal(false)}
        open={showDeploymentRequiredModal}
        title={<div className="text-lg font-medium text-orange-600">Deployment Required</div>}
        width={520}
      >
        <div className="py-4">
          {selectedModelForConfirmation && (
            <>
              <div className="flex items-center mb-4">
                <div className="bg-orange-50 p-3 rounded-lg mr-4">
                  <svg
                    fill="none"
                    height="24"
                    stroke="#f97316"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    width="24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                  </svg>
                </div>
                <div>
                  <Text strong className="text-base">
                    {selectedModelForConfirmation.training_params.job_id}
                  </Text>
                  <div className="flex items-center mt-1">
                    <Text className="text-sm text-gray-600">
                      Base Model: {selectedModelForConfirmation.training_params.base_model}
                    </Text>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                <div className="mb-3">
                  <h4 className="font-semibold text-orange-800 mb-2">Model Not Deployed</h4>
                  <p className="text-sm text-orange-700 mb-3">
                    This model has been trained successfully but is not yet deployed. 
                    You need to deploy it in the DashScope Console before you can start the service.
                  </p>
                </div>
                
                <div className="space-y-2 text-sm text-orange-700">
                  <div className="flex items-start">
                    <span className="inline-block w-6 h-6 bg-orange-200 text-orange-800 rounded-full text-xs font-bold text-center leading-6 mr-2 mt-0.5 flex-shrink-0">1</span>
                    <span>Go to the DashScope Console at bailian.console.aliyun.com</span>
                  </div>
                  <div className="flex items-start">
                    <span className="inline-block w-6 h-6 bg-orange-200 text-orange-800 rounded-full text-xs font-bold text-center leading-6 mr-2 mt-0.5 flex-shrink-0">2</span>
                    <span>Navigate to your fine-tuned models section</span>
                  </div>
                  <div className="flex items-start">
                    <span className="inline-block w-6 h-6 bg-orange-200 text-orange-800 rounded-full text-xs font-bold text-center leading-6 mr-2 mt-0.5 flex-shrink-0">3</span>
                    <span>Find and deploy model: <strong>{selectedModelForConfirmation.training_params.job_id}</strong></span>
                  </div>
                  <div className="flex items-start">
                    <span className="inline-block w-6 h-6 bg-orange-200 text-orange-800 rounded-full text-xs font-bold text-center leading-6 mr-2 mt-0.5 flex-shrink-0">4</span>
                    <span>Return here and refresh the model list to start the service</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </Modal>
    </div>
  );
}
