'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import InfoModal from '@/components/InfoModal';
import type { TrainingParams } from '@/service/train';
import { startTrain, stopTrain, retrain, getModelName, getTrainingParams } from '@/service/train';
import { useTrainingStore } from '@/store/useTrainingStore';
import { getMemoryList } from '@/service/memory';
import { message, Modal } from 'antd';
import { useModelConfigStore } from '@/store/useModelConfigStore';
import CelebrationEffect from '@/components/Celebration';
import { getModelConfig } from '@/service/modelConfig';
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

interface TrainingConfig {
  modelProvider: string;
  baseModel: string;
  modelType: string;
  epochs: number;
  learningRate: string;
  memoryPriority: string;
  showAdvanced: boolean;
}

interface TrainingDetail {
  message: string;
  timestamp: string;
}

export default function TrainingPage() {
  // Title and explanation section
  const pageTitle = 'Training Process';
  const pageDescription =
    'Transform your memories into a personalized AI model that thinks and communicates like you.';

  const [selectedInfo, setSelectedInfo] = useState<boolean>(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({} as TrainingParams);
  const [nowTrainingParams, setNowTrainingParams] = useState<TrainingParams | null>(null);
  const [trainActionLoading, setTrainActionLoading] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const firstLoadRef = useRef<boolean>(true);
  const [showCelebration, setShowCelebration] = useState(false);
  const [showMemoryModal, setShowMemoryModal] = useState(false);
  const modelConfig = useModelConfigStore((store) => store.modelConfig);
  const updateModelConfig = useModelConfigStore((store) => store.updateModelConfig);

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

  const [config, setConfig] = useState<TrainingConfig>({
    modelProvider: 'ollama',
    baseModel: 'Qwen2.5-0.5B-Instruct',
    modelType: 'General Purpose',
    epochs: 10,
    learningRate: 'Conservative (0.0001)',
    memoryPriority: 'Equal Weighting',
    showAdvanced: false
  });
  const [changeBaseModel, setChangeBaseModel] = useState(false);

  useEffect(() => {
    const nowBaseModel = JSON.parse(localStorage.getItem('trainingConfig') || '{}');

    setChangeBaseModel(nowBaseModel?.baseModel !== config.baseModel);
  }, [config.baseModel]);

  useEffect(() => {
    getModelConfig().then((res) => {
      if (res.data.code == 0) {
        const data = res.data.data || {};

        updateModelConfig(data);
      } else {
        message.error(res.data.message);
      }
    });
  }, []);

  useEffect(() => {
    getModelName().then((res) => {
      if (res.data.code === 0) {
        if (res.data.data.model_name) {
          localStorage.setItem(
            'trainingConfig',
            JSON.stringify({
              ...config,
              baseModel: res.data.data.model_name
            })
          );
        }
      }
    });
    const previousModel = localStorage.getItem('trainingConfig');

    if (previousModel) {
      setConfig({
        ...config,
        baseModel: JSON.parse(previousModel).baseModel
      });
    }
  }, []);

  const pollingInterval = useRef<any>(null);
  const router = useRouter();

  const status = useTrainingStore((state) => state.status);
  const trainingProgress = useTrainingStore((state) => state.trainingProgress);
  const [isResume, setIsResume] = useState(trainingProgress.status === 'suspended');
  const checkTrainStatus = useTrainingStore((state) => state.checkTrainStatus);
  const resetTrainingState = useTrainingStore((state) => state.resetTrainingState);
  const trainingError = useTrainingStore((state) => state.error);
  const setStatus = useTrainingStore((state) => state.setStatus);

  // Start polling training progress
  const startPolling = () => {
    // If already polling, stop first
    stopPolling();

    // Start new polling
    pollingInterval.current = setInterval(async () => {
      try {
        await checkTrainStatus();
      } catch (error) {
        console.error('Training status check failed:', error);
        stopPolling(); // Stop polling when error occurs
        setIsTraining(false);
        message.error('Training status check failed, monitoring stopped');
      }
    }, POLLING_INTERVAL);
  };

  // Stop polling
  const stopPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  };

  useEffect(() => {
    setIsResume(trainingProgress.status === 'suspended');
  }, [trainingProgress]);

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

  // Monitor training status changes, scroll to bottom when status becomes 'training'
  useEffect(() => {
    if (status === 'training') {
      scrollToBottom();
    }
  }, [status]);

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

      // Check if we were in the middle of retraining
      const isRetraining = localStorage.getItem('isRetraining') === 'true';

      if (isRetraining) {
        // If we were retraining, set status to training
        setStatus('training');
        setIsTraining(true);
        startPolling();
      }
    };

    checkMemoryCount();
  }, []);

  // Monitor training status changes and manage log connections
  useEffect(() => {
    let cleanupEventSource: (() => void) | undefined;

    // If training is in progress, start polling and establish log connection
    if (trainingProgress.status === 'in_progress') {
      startPolling();
      setIsTraining(true);

      // Create EventSource connection to get logs
      cleanupEventSource = getDetails();

      if (firstLoadRef.current) {
        scrollPageToBottom();
        scrollToBottom();
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

      // Keep EventSource open to preserve received logs
      // If resource cleanup is needed, EventSource could be closed here
    }

    // Return cleanup function to ensure EventSource is closed when component unmounts or dependencies change
    return () => {
      if (cleanupEventSource) {
        cleanupEventSource();
      }
    };
  }, [trainingProgress]);

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, []);

  const [trainingDetails, setTrainingDetails] = useState<TrainingDetail[]>([]);

  //get training params
  useEffect(() => {
    getTrainingParams()
      .then((res) => {
        if (res.data.code === 0) {
          setTrainingParams(res.data.data);
          setNowTrainingParams(res.data.data);
        } else {
          throw new Error(res.data.message);
        }
      })
      .catch((error) => {
        message.error(error.message);
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

  const scrollToBottom = () => {
    // This function is kept for backward compatibility
    // The actual scrolling is now handled by the TrainingLog component
  };

  const updateTrainingParams = (params: TrainingParams) => {
    setTrainingParams((state) => ({ ...state, ...params }));
  };

  const getDetails = () => {
    localStorage.setItem('trainingConfig', JSON.stringify(config));

    // Use EventSource to get logs
    const eventSource = new EventSource('/api/trainprocess/logs');

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        setTrainingDetails((prev) => {
          const newLogs = [
            ...prev.slice(-100),
            {
              message: data.message,
              timestamp: new Date().toISOString()
            }
          ];

          // Save logs to localStorage
          // localStorage.setItem('trainingLogs', JSON.stringify(newLogs));

          return newLogs;
        });
      } catch {
        setTrainingDetails((prev) => {
          const newLogs = [
            ...prev.slice(-100),
            {
              message: event.data,
              timestamp: new Date().toISOString()
            }
          ];

          // Save logs to localStorage
          // localStorage.setItem('trainingLogs', JSON.stringify(newLogs));

          return newLogs;
        });
      }
    };

    eventSource.onerror = (error) => {
      console.error('EventSource failed:', error);
      eventSource.close();
      message.error('Failed to get training logs');
    };

    return () => {
      eventSource.close();
    };
  };

  // Handler function for stopping training
  const handleStopTraining = async () => {
    try {
      const res = await stopTrain();

      if (res.data.code === 0) {
        setIsTraining(false);
        setIsResume(true);
      } else {
        message.error(res.data.message || 'Failed to stop training');
      }
    } catch (error) {
      console.error('Error stopping training:', error);
      message.error('Failed to stop training');
    }
  };

  // Start new training
  const handleStartNewTraining = async () => {
    setIsTraining(true);
    // Clear training logs
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    // Reset training status to initial state
    resetTrainingState();

    const apiKey = config.modelProvider === 'ollama' ? 'http://localhost:11434' : '';

    if (!apiKey) {
      setIsTraining(false);
      message.error('No API key found for the selected model');

      return;
    }

    try {
      getDetails();
      setNowTrainingParams(trainingParams);

      console.log('Using startTrain API to train new model:', config.baseModel);
      const res = await startTrain({
        model_name: config.baseModel,
        ...(isResume && !changeBaseModel ? {} : trainingParams)
      });

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('trainingConfig', JSON.stringify(config));
        setChangeBaseModel(false);
        console.log('API call successful, starting to poll for status updates');
        setStatus('training');
        scrollPageToBottom();
        startPolling();
      } else {
        message.error(res.data.message || 'Failed to start training');
        setIsTraining(false);
      }
    } catch (error: unknown) {
      console.error('Error starting training:', error);
      setIsTraining(false);
      setNowTrainingParams(null);

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
      getDetails();

      console.log('Using retrain API to retrain model:', config.baseModel);
      const res = await retrain({ model_name: config.baseModel, ...trainingParams });

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('trainingConfig', JSON.stringify(config));
        console.log('API call successful, starting to poll for status updates');
        // Set status as training to ensure UI displays correct training status
        setStatus('training');
        scrollPageToBottom();
        startPolling();
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

  // Call the appropriate handler function based on status
  const handleTrainingAction = async () => {
    if (trainActionLoading) {
      message.info('Please wait a moment...');

      return;
    }

    setTrainActionLoading(true);

    // If training is in progress, stop it
    if (isTraining) {
      await handleStopTraining();
      setTrainActionLoading(false);

      return;
    }

    // Get previously trained model information from local storage
    const previousModel = JSON.parse(localStorage.getItem('trainingConfig') || '{}');

    // If the same model has already been trained and status is 'trained' or 'running', perform retraining
    if (
      previousModel.baseModel === config.baseModel &&
      (status === 'trained' || status === 'running')
    ) {
      await handleRetrainModel();
    } else {
      // Otherwise start new training
      await handleStartNewTraining();
    }

    setTrainActionLoading(false);
  };

  const renderTrainingProgress = () => {
    return (
      <div className="space-y-6">
        {/* Training Progress Component */}
        <TrainingProgress status={status} trainingProgress={trainingProgress} />
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
          changeBaseModel={changeBaseModel}
          config={config}
          handleTrainingAction={handleTrainingAction}
          isResume={isResume}
          isTraining={isTraining}
          modelConfig={modelConfig}
          nowTrainingParams={nowTrainingParams}
          setConfig={setConfig}
          setSelectedInfo={setSelectedInfo}
          status={status}
          trainActionLoading={trainActionLoading}
          trainingParams={trainingParams}
          updateTrainingParams={updateTrainingParams}
        />

        {/* Only show training progress after training starts */}
        {(status === 'training' || status === 'trained' || status === 'running') &&
          renderTrainingProgress()}

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
      </div>
    </div>
  );
}
