'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import InfoModal from '@/components/InfoModal';
import type { TrainingConfig } from '@/service/train';
import {
  startTrain,
  stopTrain,
  retrain,
  getTrainingParams,
  checkCudaAvailability,
  resetProgress
} from '@/service/train';
import {
  startCloudTraining,
  searchJobInfo,
  getCloudTrainingStatus
} from '@/service/cloudService';
import type { TrainingJobInfo } from '@/service/cloudService';
import { useTrainingStore } from '@/store/useTrainingStore';
import { getMemoryList } from '@/service/memory';
import { message, Modal } from 'antd';
import { useModelConfigStore } from '@/store/useModelConfigStore';
import CelebrationEffect from '@/components/Celebration';
import TrainingLog from '@/components/train/TrainingLog';
import TrainingProgress from '@/components/train/TrainingProgress';
import TrainingConfiguration from '@/components/train/TrainingConfiguration';
import { ROUTER_PATH } from '@/utils/router';
interface TrainInfo {
  name: string;
  description: string;
  features: string[];
}

const trainInfo: TrainInfo = {
  name: 'Training Process',
  description:
    'Transform your memories into a personalized AI model through a multi-stage training process',
  features: [
    'Automated multi-stage training process',
    'Real-time progress monitoring',
    'Detailed training logs',
    'Training completion notification',
    'Model performance metrics'
  ]
};

const POLLING_INTERVAL = 3000;

interface TrainingDetail {
  message: string;
  timestamp: string;
}

const baseModelOptions = [
  {
    value: 'Qwen2.5-0.5B-Instruct',
    label: 'Qwen2.5-0.5B-Instruct (8GB+ RAM Recommended)'
  },
  {
    value: 'Qwen2.5-1.5B-Instruct',
    label: 'Qwen2.5-1.5B-Instruct (16GB+ RAM Recommended)'
  },
  {
    value: 'Qwen2.5-3B-Instruct',
    label: 'Qwen2.5-3B-Instruct (32GB+ RAM Recommended)'
  },
  {
    value: 'Qwen2.5-7B-Instruct',
    label: 'Qwen2.5-7B-Instruct (64GB+ RAM Recommended)'
  }
];

// Title and explanation section
const pageTitle = 'Training Process';
const pageDescription =
  'Transform your memories into a personalized AI model that thinks and communicates like you.';

export default function TrainingPage(): JSX.Element {
  const checkTrainStatus = useTrainingStore((state) => state.checkTrainStatus);
  const resetTrainingState = useTrainingStore((state) => state.resetTrainingState);
  const trainingError = useTrainingStore((state) => state.error);
  const setStatus = useTrainingStore((state) => state.setStatus);
  const fetchModelConfig = useModelConfigStore((state) => state.fetchModelConfig);
  const modelConfig = useModelConfigStore((store) => store.modelConfig);
  const status = useTrainingStore((state) => state.status);
  const trainingProgress = useTrainingStore((state) => state.trainingProgress);
  const serviceStarted = useTrainingStore((state) => state.serviceStarted);

  const router = useRouter();

  const [selectedInfo, setSelectedInfo] = useState<boolean>(false);
  const isTraining = useTrainingStore((state) => state.isTraining);
  const setIsTraining = useTrainingStore((state) => state.setIsTraining);
  const [trainingParams, setTrainingParams] = useState<TrainingConfig>({} as TrainingConfig);
  const [trainActionLoading, setTrainActionLoading] = useState(false);
  const [showCelebration, setShowCelebration] = useState(false);
  const [showMemoryModal, setShowMemoryModal] = useState(false);

  // 云端训练相关状态
  const [trainingType, setTrainingType] = useState<'local' | 'cloud'>('local');
  const [cloudJobInfo, setCloudJobInfo] = useState<TrainingJobInfo | null>(null);

  const cleanupEventSourceRef = useRef<(() => void) | undefined>();
  const containerRef = useRef<HTMLDivElement>(null);
  const firstLoadRef = useRef<boolean>(true);
  const pollingStopRef = useRef<boolean>(false);
  const cloudPollingRef = useRef<NodeJS.Timeout | null>(null); // 用于云训练轮询的引用

  const [cudaAvailable, setCudaAvailable] = useState<boolean>(false);
  const trainSuspended = useTrainingStore((state) => state.trainSuspended);
  const setTrainSuspended = useTrainingStore((state) => state.setTrainSuspended);

  useEffect(() => {
    fetchModelConfig();
  }, []);

  useEffect(() => {
    // Check CUDA availability once on load
    checkCudaAvailability()
      .then((res) => {
        if (res.data.code === 0) {
          const { cuda_available, cuda_info } = res.data.data;

          setCudaAvailable(cuda_available);

          if (cuda_available) {
            console.log('CUDA is available:', cuda_info);
          } else {
            console.log('CUDA is not available on this system');
          }
        } else {
          message.error(res.data.message || 'Failed to check CUDA availability');
        }
      })
      .catch((err) => {
        console.error('CUDA availability check failed', err);
        message.error('CUDA availability check failed');
      });
  }, []);

  // Start polling training progress
  const startPolling = () => {
    if (pollingStopRef.current) {
      return;
    }

    // Start new polling
    checkTrainStatus()
      .then(() => {
        if (pollingStopRef.current) {
          return;
        }

        setTimeout(() => {
          startPolling();
        }, POLLING_INTERVAL);
      })
      .catch((error) => {
        console.error('Training status check failed:', error);
        stopPolling(); // Stop polling when error occurs
        setIsTraining(false);
        message.error('Training status check failed, monitoring stopped');
      });
  };

  const startGetTrainingProgress = () => {
    pollingStopRef.current = false;
    setStatus('training');
    setIsTraining(true);
    startPolling();
  };

  // Stop polling
  const stopPolling = () => {
    pollingStopRef.current = true;
  };

  useEffect(() => {
    if (status === 'trained' || trainingError) {
      stopPolling();
      setIsTraining(false);

      const hasShownTrainingComplete = localStorage.getItem('hasShownTrainingComplete');

      if (hasShownTrainingComplete !== 'true' && status === 'trained' && !trainingError) {
        setTimeout(() => {
          setShowCelebration(true);
          localStorage.setItem('hasShownTrainingComplete', 'true');
        }, 1000);
      }
    }
  }, [status, trainingError]);

  // Check training status once when component loads
  useEffect(() => {
    // Check if user has at least 3 memories
    const checkMemoryCount = async () => {
      try {
        const memoryResponse = await getMemoryList();

        if (memoryResponse.data.code === 0) {
          const memories = memoryResponse.data.data;

          if (memories.length < 3) {
            // Show modal instead of direct redirect
            setShowMemoryModal(true);

            return;
          }
        }
      } catch (error) {
        console.error('Error checking memory count:', error);
      }

      // Only proceed with training status check if memory check passes
      checkTrainStatus();
    };

    checkMemoryCount();
  }, []);

  // Monitor training status changes and manage log connections
  useEffect(() => {
    // If training is in progress, start polling and establish log connection
    if (trainingProgress.status === 'in_progress') {
      setIsTraining(true);

      if (firstLoadRef.current) {
        scrollPageToBottom();

        // On first load, start polling and get training progress.
        startGetTrainingProgress();
      }
    }
    // If training is completed or failed, stop polling
    else if (
      trainingProgress.status === 'completed' ||
      trainingProgress.status === 'failed' ||
      trainingProgress.status === 'suspended'
    ) {
      stopPolling();
      setIsTraining(false);
    }
  }, [trainingProgress]);

  useEffect(() => {
    if (isTraining) {
      updateTrainLog();
    }
  }, [isTraining]);

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      stopPolling(); // 停止本地训练轮询
      stopCloudPolling(); // 停止云端训练轮询
    };
  }, []);

  const [trainingDetails, setTrainingDetails] = useState<TrainingDetail[]>([]);

  //get training params
  useEffect(() => {
    getTrainingParams()
      .then((res) => {
        if (res.data.code === 0) {
          const data = res.data.data;

          setTrainingParams(data);

          localStorage.setItem('trainingParams', JSON.stringify(data));
        } else {
          throw new Error(res.data.message);
        }
      })
      .catch((error) => {
        console.error(error.message);
      });
  }, []);

  useEffect(() => {
    const savedLogs = localStorage.getItem('trainingLogs');

    setTrainingDetails(savedLogs ? JSON.parse(savedLogs) : []);
  }, []);

  // Scroll to the bottom of the page
  const scrollPageToBottom = () => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: 'smooth'
    });
    // Set that it's no longer the first load
    firstLoadRef.current = false;
  };

  const updateTrainingParams = (params: TrainingConfig) => {
    setTrainingParams((state: TrainingConfig) => ({ ...state, ...params }));
  };

  const getDetails = () => {
    // Use EventSource to get logs
    const eventSource = new EventSource(`/api/trainprocess/logs`);

    eventSource.onmessage = (event) => {
      // Don't try to parse as JSON, just use the raw text data directly
      const logMessage = event.data;

      setTrainingDetails((prev) => {
        const newLogs = [
          ...prev.slice(-500), // Keep more log entries (500 instead of 100)
          {
            message: logMessage,
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            })
          }
        ];

        // Save logs to localStorage for persistence between page refreshes
        localStorage.setItem('trainingLogs', JSON.stringify(newLogs));

        return newLogs;
      });
    };

    eventSource.onerror = (error) => {
      console.error('EventSource failed:', error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  };

  const updateTrainLog = () => {
    if (cleanupEventSourceRef.current) {
      cleanupEventSourceRef.current();
    }

    cleanupEventSourceRef.current = getDetails();
  };

  // Handler function for stopping training
  const handleStopTraining = async () => {
    try {
      const res = await stopTrain();

      if (res.data.code === 0) {
        setIsTraining(false);
        setTrainSuspended(true);
      } else {
        message.error(res.data.message || 'Failed to stop training');
      }
    } catch (error) {
      console.error('Error stopping training:', error);
      message.error('Failed to stop training');
    }
  };

  const handleResetProgress = () => {
    setTrainActionLoading(true);

    resetProgress()
      .then((res) => {
        if (res.data.code === 0) {
          setTrainSuspended(false);
          resetTrainingState();
          localStorage.removeItem('trainingLogs');
        } else {
          throw new Error(res.data.message || 'Failed to reset progress');
        }
      })
      .catch((error) => {
        console.error('Error resetting progress:', error);
      })
      .finally(() => {
        setTrainActionLoading(false);
      });
  };

  // Start new training
  const handleStartNewTraining = async () => {
    setIsTraining(true);
    // Clear training logs
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    // Reset training status to initial state
    resetTrainingState();

    try {
      console.log('Using startTrain API to train new model:', trainingParams.model_name);
      const res = await startTrain({
        ...trainingParams,
        model_name: trainingParams.local_model_name
      });

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('trainingParams', JSON.stringify(trainingParams));
        scrollPageToBottom();
        startGetTrainingProgress();
      } else {
        message.error(res.data.message || 'Failed to start training');
        setIsTraining(false);
      }
    } catch (error: unknown) {
      console.error('Error starting training:', error);
      setIsTraining(false);

      if (error instanceof Error) {
        message.error(error.message || 'Failed to start training');
      } else {
        message.error('Failed to start training');
      }
    }
  };

  // Retrain existing model
  const handleRetrainModel = async () => {
    setIsTraining(true);
    // Clear training logs
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    // Reset training status to initial state
    resetTrainingState();

    try {
      const res = await retrain(trainingParams);

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('trainingParams', JSON.stringify(trainingParams));
        scrollPageToBottom();
        startGetTrainingProgress();
      } else {
        message.error(res.data.message || 'Failed to retrain model');
        setIsTraining(false);
      }
    } catch (error: unknown) {
      console.error('Error retraining model:', error);
      setIsTraining(false);

      if (error instanceof Error) {
        message.error(error.message || 'Failed to retrain model');
      } else {
        message.error('Failed to retrain model');
      }
    }
  };

  // 统一处理本地和云端训练的动作
  const handleTrainingAction = async (type: 'local' | 'cloud' = 'local') => {
    if (trainActionLoading) {
      message.info('Please wait a moment...');

      return;
    }

    if (!isTraining && serviceStarted) {
      message.error('Model is already running, please stop it first');

      return;
    }

    setTrainActionLoading(true);

    // 更新当前活动的训练类型
    setTrainingType(type);

    // If training is in progress, stop it
    if (isTraining) {
      if (type === 'cloud') {
        stopCloudPolling();
        setIsTraining(false);
      } else {
        await handleStopTraining();
      }

      setTrainActionLoading(false);

      return;
    }

    // 根据训练类型选择不同的处理方法
    if (type === 'cloud') {
      // 云端训练流程
      await handleStartCloudTraining();
    } else {
      // 本地训练流程
      if (status === 'trained') {
        await handleRetrainModel();
      } else {
        await handleStartNewTraining();
      }
    }

    setTrainActionLoading(false);
  };

  // 云端训练相关方法
  // 启动云端训练
  const handleStartCloudTraining = async () => {
    setIsTraining(true);
    // 清除训练日志
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    // 重置训练状态
    resetTrainingState();

    try {
      const res = await startCloudTraining({
        base_model: trainingParams.cloud_model_name,
        training_type: 'efficient_sft',
        hyper_parameters: {
          n_epochs: trainingParams.number_of_epochs,
          learning_rate: trainingParams.learning_rate
        }
      });

      if (res.data.code === 0) {
        // 保存训练配置并开始轮询
        localStorage.setItem('trainingParams', JSON.stringify(trainingParams));
        scrollPageToBottom();
        setTrainingType('cloud');
        startCloudTrainingPolling();
      } else {
        message.error(res.data.message || 'Failed to start cloud training');
        setIsTraining(false);
      }
    } catch (error) {
      console.error('Error starting cloud training:', error);
      setIsTraining(false);

      if (error instanceof Error) {
        message.error(error.message || 'Failed to start cloud training');
      } else {
        message.error('Failed to start cloud training');
      }
    }
  };

  // 开始云端训练轮询
  const startCloudTrainingPolling = () => {
    // 设置状态为训练中
    setStatus('training');
    setIsTraining(true);

    // 先轮询任务信息获取 job_id
    pollCloudJobInfo();
  };

  // 轮询云端任务信息
  const pollCloudJobInfo = () => {
    if (pollingStopRef.current) return;

    searchJobInfo()
      .then((res) => {
        if (res.data.code === 0) {
          const jobInfo = res.data.data as TrainingJobInfo;

          setCloudJobInfo(jobInfo);

          if (jobInfo && jobInfo.job_id) {
            // 如果获取到了 job_id，就开始轮询训练状态
            pollCloudTrainingStatus(jobInfo.job_id);

            return;
          }
        }

        // 如果没有获取到 job_id，继续轮询
        if (!pollingStopRef.current) {
          cloudPollingRef.current = setTimeout(() => {
            pollCloudJobInfo();
          }, POLLING_INTERVAL);
        }
      })
      .catch((error) => {
        console.error('Error polling cloud job info:', error);

        if (!pollingStopRef.current) {
          cloudPollingRef.current = setTimeout(() => {
            pollCloudJobInfo();
          }, POLLING_INTERVAL);
        }
      });
  };

  // 根据 job_id 轮询云端训练状态
  const pollCloudTrainingStatus = (jobId: string) => {
    if (pollingStopRef.current) return;

    getCloudTrainingStatus(jobId)
      .then((res) => {
        if (res.data.code === 0) {
          const jobStatus = res.data.message;

          // 根据状态更新进度
          if (jobStatus === 'SUCCEEDED') {
            // 训练完成
            setStatus('trained');
            setIsTraining(false);

            // 显示训练完成庆祝效果
            const hasShownTrainingComplete = localStorage.getItem('hasShownTrainingComplete');

            if (hasShownTrainingComplete !== 'true') {
              setTimeout(() => {
                setShowCelebration(true);
                localStorage.setItem('hasShownTrainingComplete', 'true');
              }, 1000);
            }

            stopCloudPolling();

            return;
          } else if (jobStatus === 'FAILED') {
            // 训练失败
            message.error('Cloud training failed');
            setIsTraining(false);
            stopCloudPolling();
            return;
          }

          // 继续轮询
          if (!pollingStopRef.current) {
            cloudPollingRef.current = setTimeout(() => {
              pollCloudTrainingStatus(jobId);
            }, POLLING_INTERVAL);
          }
        } else {
          // API 错误
          message.error(res.data.message || 'Failed to get cloud training status');

          if (!pollingStopRef.current) {
            cloudPollingRef.current = setTimeout(() => {
              pollCloudTrainingStatus(jobId);
            }, POLLING_INTERVAL);
          }
        }
      })
      .catch((error) => {
        console.error('Error polling cloud training status:', error);

        if (!pollingStopRef.current) {
          cloudPollingRef.current = setTimeout(() => {
            pollCloudTrainingStatus(jobId);
          }, POLLING_INTERVAL);
        }
      });
  };

  // 停止云端轮询
  const stopCloudPolling = () => {
    pollingStopRef.current = true;

    if (cloudPollingRef.current) {
      clearTimeout(cloudPollingRef.current);
      cloudPollingRef.current = null;
    }
  };

  // 清理资源
  useEffect(() => {
    return () => {
      stopPolling();
      stopCloudPolling();
    };
  }, []);

  const renderTrainingProgress = () => {
    return (
      <div className="space-y-6">
        {/* Training Progress Component */}
        <TrainingProgress 
          status={status} 
          trainingProgress={trainingProgress}
          trainingType={trainingType}
          cloudJobInfo={cloudJobInfo}
        />
      </div>
    );
  };

  const renderTrainingLog = () => {
    return (
      <div className="space-y-6">
        {/* Training Log Console */}
        <TrainingLog trainingDetails={trainingDetails} />
      </div>
    );
  };

  // Handle memory modal confirmation
  const handleMemoryModalConfirm = () => {
    setShowMemoryModal(false);
    router.push(ROUTER_PATH.TRAIN_MEMORIES);
  };

  return (
    <div ref={containerRef} className="h-full overflow-auto">
      {/* Memory count warning modal */}
      <Modal
        cancelText="Stay Here"
        okText="Go to Memories Page"
        onCancel={() => setShowMemoryModal(false)}
        onOk={handleMemoryModalConfirm}
        open={showMemoryModal}
        title="More Memories Needed"
      >
        <p>You need to add at least 3 memories before you can train your model.</p>
        <p>Would you like to go to the memories page to add more?</p>
      </Modal>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Page Title and Description */}
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{pageTitle}</h1>
          <p className="text-gray-600 max-w-3xl">{pageDescription}</p>
        </div>
        {/* Training Configuration Component */}
        <TrainingConfiguration
          baseModelOptions={baseModelOptions}
          cudaAvailable={cudaAvailable}
          handleResetProgress={handleResetProgress}
          handleTrainingAction={handleTrainingAction}
          isTraining={isTraining}
          modelConfig={modelConfig}
          setSelectedInfo={setSelectedInfo}
          status={status}
          trainActionLoading={trainActionLoading}
          trainSuspended={trainSuspended}
          trainingParams={trainingParams}
          updateTrainingParams={updateTrainingParams}
          trainingType={trainingType}
          setTrainingType={setTrainingType}
        />

        {/* Only show training progress after training starts */}
        {(status === 'training' || status === 'trained') && renderTrainingProgress()}

        {/* Always show training log regardless of training status */}
        {renderTrainingLog()}

        {/* L1 and L2 Panels - show when training is complete or model is running */}

        <InfoModal
          content={
            <div className="space-y-4">
              <p className="text-gray-600">{trainInfo.description}</p>
              <div>
                <h4 className="font-medium mb-2">Key Features:</h4>
                <ul className="list-disc pl-5 space-y-1.5">
                  {trainInfo.features.map((feature, index) => (
                    <li key={index} className="text-gray-600">
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          }
          onClose={() => setSelectedInfo(false)}
          open={!!selectedInfo && !!trainInfo}
          title={trainInfo.name}
        />

        {/* Training completion celebration effect */}
        <CelebrationEffect isVisible={showCelebration} onClose={() => setShowCelebration(false)} />

        {/* 调试区域：显示Training参数 */}
        <div className="mt-8 p-4 border rounded-md bg-gray-50">
          <h3 className="text-lg font-medium mb-4">训练参数调试信息</h3>
          <div className="bg-white p-3 rounded shadow-inner overflow-auto max-h-80">
            <pre className="text-sm text-gray-700">{JSON.stringify(trainingParams, null, 2)}</pre>
          </div>
          <div className="mt-4">
            <h4 className="font-medium mb-2">当前训练类型:</h4>
            <div className="bg-white p-3 rounded shadow-inner">
              <span className="font-mono">{trainingType}</span>
            </div>
          </div>
          {cloudJobInfo && (
            <div className="mt-4">
              <h4 className="font-medium mb-2">云训练任务信息:</h4>
              <div className="bg-white p-3 rounded shadow-inner">
                <pre className="text-sm text-gray-700">{JSON.stringify(cloudJobInfo, null, 2)}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
