'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import InfoModal from '@/components/InfoModal';
import type { TrainingConfig, LocalTrainingParams, CloudTrainingParams, TrainingParamsResponse } from '@/service/train';
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
  getCloudTrainingProgress,
  stopCloudTraining,
  resetCloudTrainingProgress,
  resumeCloudTraining
} from '@/service/cloudService';
import type { CloudProgressData } from '@/service/cloudService';
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
    value: 'Qwen3-0.6B',
    label: 'Qwen3-0.6B (8GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-1.7B',
    label: 'Qwen3-1.7B (16GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-4B',
    label: 'Qwen3-4B (32GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-8B',
    label: 'Qwen3-8B (64GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-14B',
    label: 'Qwen3-14B (96GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-32B',
    label: 'Qwen3-32B (192GB+ RAM Recommended)'
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
  const [localTrainingParams, setLocalTrainingParams] = useState<LocalTrainingParams>({} as LocalTrainingParams);
  const [cloudTrainingParams, setCloudTrainingParams] = useState<CloudTrainingParams>({} as CloudTrainingParams);
  const [trainActionLoading, setTrainActionLoading] = useState(false);
  const [showCelebration, setShowCelebration] = useState(false);
  const [showMemoryModal, setShowMemoryModal] = useState(false);

  const [trainingType, setTrainingType] = useState<'local' | 'cloud'>('local');
  const [cloudProgress, setCloudProgress] = useState<CloudProgressData | null>(null);
  const [cloudJobId, setCloudJobId] = useState<string | null>(null);
  const [cloudTrainSuspended, setCloudTrainSuspended] = useState(false);
  const [cloudTrainingStatus, setCloudTrainingStatus] = useState<'idle' | 'training' | 'trained' | 'failed' | 'suspended'>('idle');
  // Track pause request state to avoid state inconsistency
  const [isPauseRequested, setIsPauseRequested] = useState(false);
  // Track pause status polling
  const [pauseStatus, setPauseStatus] = useState<'success' | 'pending' | 'failed' | null>(null);
  const pausePollingRef = useRef<NodeJS.Timeout | null>(null);
  // Track polling retry attempts
  const [pollingRetryCount, setPollingRetryCount] = useState(0);
  const maxPollingRetries = 3;
  // Pause timeout detection
  const pauseTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pauseTimeoutDuration = 30000; // 30 seconds timeout for pause operations

  const cleanupEventSourceRef = useRef<(() => void) | undefined>();
  const containerRef = useRef<HTMLDivElement>(null);
  const firstLoadRef = useRef<boolean>(true);
  const pollingStopRef = useRef<boolean>(false);
  const cloudPollingRef = useRef<NodeJS.Timeout | null>(null); 

  const [cudaAvailable, setCudaAvailable] = useState<boolean>(false);
  const trainSuspended = useTrainingStore((state) => state.trainSuspended);
  const setTrainSuspended = useTrainingStore((state) => state.setTrainSuspended);

  const startCloudTrainingPolling = () => {
    setStatus('training');
    setIsTraining(true);
    pollingStopRef.current = false; // Reset polling stop flag
    pollCloudProgress();
  };

  // New function to poll cloud training progress
  const pollCloudProgress = () => {
    if (pollingStopRef.current) return;

    getCloudTrainingProgress()
      .then((res) => {
        if (pollingStopRef.current) return; // Check again in case it was stopped during API call

        if (res.data.code === 0) {
          const progressData = res.data.data.progress;
          const currentJobId = res.data.data.job_id;

          setCloudProgress(progressData);
          if (currentJobId) {
            setCloudJobId(currentJobId);
          }
          
          // Handle different training statuses
          if (progressData.status === 'completed') {
            // Training completed successfully
            setStatus('trained');
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('trained');
            
            // Show celebration effect
            const hasShownTrainingComplete = localStorage.getItem('hasShownCloudTrainingComplete');
            if (hasShownTrainingComplete !== 'true') {
              setTimeout(() => {
                setShowCelebration(true);
                localStorage.setItem('hasShownCloudTrainingComplete', 'true');
              }, 1000);
            }
            stopCloudPolling();
            return;
          } else if (progressData.status === 'failed') {
            // Training failed
            setStatus('training');
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('failed');
            stopCloudPolling();
            return;
          } else if (progressData.status === 'suspended') {
            // Training suspended - this should normally be handled by the pause API response
            // But in case we detect it here, update the state accordingly
            setStatus('training');
            setIsTraining(false);
            setCloudTrainSuspended(true);
            setCloudTrainingStatus('suspended');
            
            // Clear pause-related states if they exist
            if (pauseTimeoutRef.current) {
              clearTimeout(pauseTimeoutRef.current);
              pauseTimeoutRef.current = null;
            }
            setIsPauseRequested(false);
            
            message.info('Cloud training is paused.');
            stopCloudPolling();
            return;
          } else if (progressData.status === 'in_progress') {
            // Training is running
            setStatus('training');
            setIsTraining(true);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('training');
          }

          // Continue polling if still in progress
          if (!pollingStopRef.current) {
            setPollingRetryCount(0); // Reset retry count on success
            cloudPollingRef.current = setTimeout(pollCloudProgress, POLLING_INTERVAL);
          }
        } else {
          // Handle API errors with intelligent retry
          if (pollingRetryCount < maxPollingRetries) {
            setPollingRetryCount(prev => prev + 1);
            message.warning(`Cloud training status check failed (${pollingRetryCount + 1}/${maxPollingRetries}). Retrying...`);
            
            if (!pollingStopRef.current) {
              // Exponential backoff for retries
              const retryDelay = POLLING_INTERVAL * Math.pow(2, pollingRetryCount);
              cloudPollingRef.current = setTimeout(pollCloudProgress, retryDelay);
            }
          } else {
            message.error(res.data.message || 'Failed to get cloud training progress after multiple retries');
            setPollingRetryCount(0);
            stopCloudPolling();
          }
        }
      })
      .catch((error) => {
        console.error('Error polling cloud training progress:', error);
        
        // Handle network errors with intelligent retry
        if (pollingRetryCount < maxPollingRetries) {
          setPollingRetryCount(prev => prev + 1);
          message.warning(`Network error checking cloud training status (${pollingRetryCount + 1}/${maxPollingRetries}). Retrying...`);
          
          if (!pollingStopRef.current) {
            // Exponential backoff for retries
            const retryDelay = POLLING_INTERVAL * Math.pow(2, pollingRetryCount);
            cloudPollingRef.current = setTimeout(pollCloudProgress, retryDelay);
          }
        } else {
          message.error('Unable to check cloud training status after multiple retries. Please refresh the page.');
          setPollingRetryCount(0);
          stopCloudPolling();
        }
      });
  };

  const stopCloudPolling = () => {
    pollingStopRef.current = true;

    if (cloudPollingRef.current) {
      clearTimeout(cloudPollingRef.current);
      cloudPollingRef.current = null;
    }
  };

  // Poll pause status until it's no longer pending
  const pollPauseStatus = () => {
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
    }

    pausePollingRef.current = setTimeout(async () => {
      try {
        const res = await stopCloudTraining();
        
        if (res.data.code === 0 && res.data.data) {
          const status = res.data.data.status;
          setPauseStatus(status);
          
          if (status === 'pending') {
            // Continue polling if still pending
            pollPauseStatus();
          } else if (status === 'success') {
            // Successfully stopped
            setIsTraining(false);
            setCloudTrainSuspended(true);
            setCloudTrainingStatus('suspended');
            setIsPauseRequested(false);
            setPauseStatus(null);
            stopCloudPolling();
            message.success('Cloud training stopped successfully');
          } else if (status === 'failed') {
            // Failed to stop
            setIsPauseRequested(false);
            setPauseStatus(null);
            message.error('Failed to stop cloud training');
          }
        } else {
          // API error
          setIsPauseRequested(false);
          setPauseStatus(null);
          message.error(res.data.message || 'Failed to check pause status');
        }
      } catch (error) {
        console.error('Error polling pause status:', error);
        setIsPauseRequested(false);
        setPauseStatus(null);
        message.error('Error checking pause status');
      }
    }, 10000); // Poll every 10 seconds
  };

  // Check cloud training pause status on page load
  const checkCloudPauseStatus = async (): Promise<'pending' | 'success' | 'failed' | null> => {
    try {
      const res = await stopCloudTraining();
      
      if (res.data.code === 0 && res.data.data) {
        const status = res.data.data.status;
        
        if (status === 'pending') {
          // If pause is still pending, show the pending state and start polling
          setIsPauseRequested(true);
          setPauseStatus('pending');
          pollPauseStatus();
          console.log('Found pending pause operation, starting status polling...');
          return 'pending';
        } else if (status === 'success') {
          // Pause was completed successfully - update UI to show suspended state
          setIsTraining(false);
          setCloudTrainSuspended(true);
          setCloudTrainingStatus('suspended');
          setIsPauseRequested(false);
          setPauseStatus(null);
          stopCloudPolling();
          console.log('Previous pause operation completed successfully, UI updated to suspended state');
          return 'success';
        } else if (status === 'failed') {
          // Pause failed, clear any pending state
          setIsPauseRequested(false);
          setPauseStatus(null);
          console.log('Previous pause operation failed');
          return 'failed';
        }
      }
      return null;
    } catch (error) {
      // Silently handle errors - this is just a status check
      console.log('No pending pause operation found or error checking pause status:', error);
      return null;
    }
  };

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

  // Check cloud training progress on page load
  const checkCloudTrainingProgress = async () => {
    try {
      const res = await getCloudTrainingProgress();
      
      if (res.data.code === 0) {
        const progressData = res.data.data.progress;
        const currentJobId = res.data.data.job_id;
        
        if (progressData) {
          setCloudProgress(progressData);
          
          if (currentJobId) {
            setCloudJobId(currentJobId);
          }

          // If training is in progress, check for pause status first
          if (progressData.status === 'in_progress') {
            setTrainingType('cloud');
            console.log('Cloud training detected as in_progress, checking pause status...');
            
            // Check if there's a pending or completed pause operation first
            const pauseStatus = await checkCloudPauseStatus();
            console.log('Pause status check result:', pauseStatus);
            
            // If pause was successful or pending, don't start training flow
            if (pauseStatus !== 'success' && pauseStatus !== 'pending') {
              console.log('No active pause found, starting normal training flow...');
              setIsTraining(true);
              setStatus('training');
              setCloudTrainSuspended(false);
              setCloudTrainingStatus('training');
              startCloudTrainingPolling();
            } else {
              console.log('Pause operation found, skipping training flow start');
            }
          } else if (progressData.status === 'completed') {
            setTrainingType('cloud');
            setStatus('trained');
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('trained');
          } else if (progressData.status === 'failed') {
            setTrainingType('cloud');
            setStatus('training'); // Keep as 'training' since ModelStatus doesn't have 'failed'
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('failed');
          } else if (progressData.status === 'suspended') {
            setTrainingType('cloud');
            setStatus('training');
            setIsTraining(false);
            setCloudTrainSuspended(true);
            setCloudTrainingStatus('suspended');
          }
        }
      }
    } catch (error) {
      // Silently handle errors for initial check - cloud training might not be active
      console.log('No active cloud training found or error checking cloud progress:', error);
    }
  };

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

      // Check both local and cloud training status
      await Promise.allSettled([
        checkTrainStatus(),
        checkCloudTrainingProgress()
      ]);
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
      stopPolling(); 
      if (cloudPollingRef.current) { // Ensure cloud polling is stopped
        clearTimeout(cloudPollingRef.current);
      }
      if (pauseTimeoutRef.current) { // Clear pause timeout
        clearTimeout(pauseTimeoutRef.current);
      }
      if (pausePollingRef.current) { // Clear pause polling
        clearTimeout(pausePollingRef.current);
      }
      pollingStopRef.current = true; // Set flag to stop any ongoing polling
    };
  }, []);

  const [trainingDetails, setTrainingDetails] = useState<TrainingDetail[]>([]);

  //get training params
  useEffect(() => {
    getTrainingParams()
      .then((res) => {
        if (res.data.code === 0) {
          const data = res.data.data;

          // Set separate local and cloud training params
          setLocalTrainingParams(data.local);
          setCloudTrainingParams(data.cloud);

          localStorage.setItem('localTrainingParams', JSON.stringify(data.local));
          localStorage.setItem('cloudTrainingParams', JSON.stringify(data.cloud));
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

  const updateLocalTrainingParams = (params: Partial<LocalTrainingParams>) => {
    setLocalTrainingParams((state: LocalTrainingParams) => ({ ...state, ...params }));
  };

  const updateCloudTrainingParams = (params: Partial<CloudTrainingParams>) => {
    setCloudTrainingParams((state: CloudTrainingParams) => ({ ...state, ...params }));
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

  // Helper function to check if cloud training is in the critical stage
  const isInCriticalStage = (progressData: CloudProgressData | null): boolean => {
    if (!progressData) return false;
    
    // Check if we're in the "Training to create Second Me" stage
    const currentStage = progressData.current_stage;
    
    // Check by current_stage field
    if (currentStage === 'training_to_create_second_me') {
      return true;
    }
    
    // Also check if any stage has the "Training to create Second Me" name and is in progress
    if (progressData.stages) {
      return progressData.stages.some(stage => 
        stage.name === 'Training to create Second Me' && 
        stage.status === 'in_progress'
      );
    }
    
    return false;
  };

  // Handle cloud training stop with confirmation for critical stage
  const handleStopCloudTraining = async (): Promise<boolean> => {
    // Prevent multiple pause requests
    if (isPauseRequested) {
      message.warning('Pause request already in progress, please wait...');
      return false;
    }

    // Check if we're in the critical final stage
    if (isInCriticalStage(cloudProgress)) {
      return new Promise((resolve) => {
        Modal.confirm({
          title: 'Cloud Training Critical Stage Warning',
          content: (
            <div className="space-y-3">
              <p className="text-amber-600 font-medium">
                ⚠️ You are currently in the &quot;Training to create Second Me&quot; stage of cloud training.
              </p>
              <p>
                This critical stage does not support checkpoint resume functionality. If you stop cloud training now:
              </p>
              <ul className="list-disc pl-6 space-y-1">
                <li>All cloud training progress will be lost</li>
                <li>Alibaba Cloud training costs are non-refundable</li>
                <li>You will need to restart the entire cloud training process</li>
              </ul>
              <p className="font-medium">Are you sure you want to stop cloud training?</p>
            </div>
          ),
          okText: 'Yes, Stop Training',
          cancelText: 'Cancel',
          okType: 'danger',
          onOk: async () => {
            setIsPauseRequested(true);
            try {
              const res = await stopCloudTraining();
              
              if (res.data.code === 0 && res.data.data) {
                const status = res.data.data.status;
                setPauseStatus(status);
                
                if (status === 'pending') {
                  // Start polling for status updates
                  pollPauseStatus();
                  resolve(true);
                } else if (status === 'success') {
                  // Immediately update state when pause is successful
                  setIsTraining(false);
                  setCloudTrainSuspended(true);
                  setCloudTrainingStatus('suspended');
                  setIsPauseRequested(false);
                  setPauseStatus(null);
                  stopCloudPolling();
                  message.success('Cloud training stopped successfully');
                  resolve(true);
                } else { // 'failed'
                  message.error('Failed to stop cloud training');
                  setIsPauseRequested(false);
                  setPauseStatus(null);
                  resolve(false);
                }
              } else {
                message.error(res.data.message || 'Failed to stop cloud training');
                setIsPauseRequested(false);
                setPauseStatus(null);
                resolve(false);
              }
            } catch (error) {
              console.error('Error stopping cloud training:', error);
              message.error('Failed to stop cloud training');
              setIsPauseRequested(false);
              setPauseStatus(null);
              resolve(false);
            }
          },
          onCancel: () => resolve(false)
        });
      });
    }

    // Normal stopping for non-critical stages (supports checkpoint resume)
    setIsPauseRequested(true);
    try {
      const res = await stopCloudTraining();
      
      if (res.data.code === 0 && res.data.data) {
        const status = res.data.data.status;
        setPauseStatus(status);
        
        if (status === 'pending') {
          // Start polling for status updates
          pollPauseStatus();
          return true;
        } else if (status === 'success') {
          // Immediately update state when pause is successful
          setIsTraining(false);
          setCloudTrainSuspended(true);
          setCloudTrainingStatus('suspended');
          setIsPauseRequested(false);
          setPauseStatus(null);
          
          // Clear pause timeout if it exists
          if (pauseTimeoutRef.current) {
            clearTimeout(pauseTimeoutRef.current);
            pauseTimeoutRef.current = null;
          }
          
          // Stop polling immediately since pause is confirmed
          stopCloudPolling();
          message.success('Cloud training stopped successfully');
          
          return true;
        } else { // 'failed'
          message.error('Failed to stop cloud training');
          setIsPauseRequested(false);
          setPauseStatus(null);
          return false;
        }
      }
      
      message.error(res.data.message || 'Failed to stop cloud training');
      setIsPauseRequested(false);
      setPauseStatus(null);
      
      return false;
    } catch (error) {
      console.error('Error stopping cloud training:', error);
      message.error('Failed to stop cloud training');
      setIsPauseRequested(false);
      setPauseStatus(null);
      
      return false;
    }
  };

  // Handle cloud training reset
  const handleResetCloudProgress = async () => {
    setTrainActionLoading(true);

    try {
      const res = await resetCloudTrainingProgress();
      if (res.data.code === 0) {
        setCloudTrainSuspended(false);
        setCloudProgress(null);
        setCloudJobId(null);
        setCloudTrainingStatus('idle');
        setStatus('training');
        resetTrainingState();
        
        // Clear pause state
        setIsPauseRequested(false);
        setPauseStatus(null);
        if (pausePollingRef.current) {
          clearTimeout(pausePollingRef.current);
          pausePollingRef.current = null;
        }
        
        localStorage.removeItem('trainingLogs');
        localStorage.removeItem('hasShownCloudTrainingComplete');
        
        // Request progress after reset to get the initialized progress state
        try {
          const progressRes = await getCloudTrainingProgress();
          if (progressRes.data.code === 0) {
            const progressData = progressRes.data.data.progress;
            if (progressData) {
              setCloudProgress(progressData);
              // Update job ID if available
              if (progressRes.data.data.job_id) {
                setCloudJobId(progressRes.data.data.job_id);
              }
              console.log('Retrieved initial cloud training progress after reset:', progressData);
            }
          } else {
            console.log('No initial progress found after reset, which is expected for a fresh start');
          }
        } catch (progressError) {
          console.error('Error getting initial progress after reset:', progressError);
          // This is not a critical error, just log it
        }
      } else {
        message.error(res.data.message || 'Failed to reset cloud training progress');
      }
    } catch (error) {
      console.error('Error resetting cloud training progress:', error);
      message.error('Failed to reset cloud training progress');
    } finally {
      setTrainActionLoading(false);
    }
  };

  // Handle resume cloud training
  const handleResumeCloudTraining = async () => {
    setIsTraining(true);
    setCloudTrainSuspended(false);
    setCloudTrainingStatus('training');
    
    // Clear pause state
    setIsPauseRequested(false);
    setPauseStatus(null);
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
      pausePollingRef.current = null;
    }
    
    try {
      // Call the resume API endpoint
      const response = await resumeCloudTraining();
      
      if (response.data.code === 0) {
        // Resume successful, restart polling
        setStatus('training');
        startCloudTrainingPolling();
      } else {
        throw new Error(response.data.message || 'Failed to resume cloud training');
      }
    } catch (error) {
      console.error('Error resuming cloud training:', error);
      setIsTraining(false);
      setCloudTrainSuspended(true);
      setCloudTrainingStatus('suspended');
      message.error('Failed to resume cloud training');
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
      console.log('Using startTrain API to train new model:', localTrainingParams.model_name);
      const res = await startTrain({
        ...localTrainingParams,
        // Convert to legacy format for compatibility
        local_model_name: localTrainingParams.model_name,
        cloud_model_name: cloudTrainingParams.model_name
      } as TrainingConfig);

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('localTrainingParams', JSON.stringify(localTrainingParams));
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
      const res = await retrain({
        ...localTrainingParams,
        // Convert to legacy format for compatibility
        local_model_name: localTrainingParams.model_name,
        cloud_model_name: cloudTrainingParams.model_name
      } as TrainingConfig);

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('localTrainingParams', JSON.stringify(localTrainingParams));
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

    setTrainingType(type);

    // If training is in progress, stop it
    if (isTraining) {
      if (type === 'cloud') {
        await handleStopCloudTraining();
        // handleStopCloudTraining now handles state updates and polling stop immediately
        setTrainActionLoading(false);
        return;
      } else {
        await handleStopTraining();
      }

      setTrainActionLoading(false);

      return;
    }

    if (type === 'cloud') {
      if (cloudTrainSuspended) {
        // Resume cloud training
        await handleResumeCloudTraining();
      } else if (cloudTrainingStatus === 'trained') {
        // Retrain cloud model
        await handleStartCloudTraining();
      } else if (cloudTrainingStatus === 'failed') {
        // Retry or resume failed training
        await handleStartCloudTraining();
      } else {
        // Start new cloud training
        await handleStartCloudTraining();
      }
    } else {
      if (status === 'trained') {
        await handleRetrainModel();
      } else {
        await handleStartNewTraining();
      }
    }

    setTrainActionLoading(false);
  };

  const handleStartCloudTraining = async () => {
    setIsTraining(true);
    setStatus('training');
    setCloudTrainingStatus('training');
    
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    localStorage.removeItem('hasShownCloudTrainingComplete'); // Reset celebration flag
    
    resetTrainingState();
    setCloudProgress(null); // Reset cloud progress
    setCloudJobId(null); // Reset cloud job ID
    setCloudTrainSuspended(false); // Reset suspension state
    
    // Clear pause state
    setIsPauseRequested(false);
    setPauseStatus(null);
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
      pausePollingRef.current = null;
    }

    try {
      const res = await startCloudTraining({
        base_model: cloudTrainingParams.base_model,
        training_type: cloudTrainingParams.training_type || 'efficient_sft',
        data_synthesis_mode: cloudTrainingParams.data_synthesis_mode || 'medium',
        language: cloudTrainingParams.language || 'english',
        hyper_parameters: {
          n_epochs: cloudTrainingParams.hyper_parameters?.n_epochs,
          learning_rate: cloudTrainingParams.hyper_parameters?.learning_rate
        }
      });

      if (res.data.code === 0) {
        const data = res.data.data as { job_id?: string };
        const returnedJobId = data.job_id;
        if (returnedJobId) {
          setCloudJobId(returnedJobId);
        }
        
        localStorage.setItem('cloudTrainingParams', JSON.stringify(cloudTrainingParams));
        scrollPageToBottom();
        setTrainingType('cloud');
        startCloudTrainingPolling();
      } else {
        message.error(res.data.message || 'Failed to start cloud training');
        setIsTraining(false);
        setCloudTrainingStatus('idle');
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
          cloudProgressData={cloudProgress}
          cloudJobId={cloudJobId}
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
          handleResetProgress={trainingType === 'cloud' ? handleResetCloudProgress : handleResetProgress}
          handleTrainingAction={handleTrainingAction}
          isTraining={isTraining}
          modelConfig={modelConfig}
          setSelectedInfo={setSelectedInfo}
          status={status}
          trainActionLoading={trainActionLoading}
          trainSuspended={trainingType === 'cloud' ? cloudTrainSuspended : trainSuspended}
          localTrainingParams={localTrainingParams}
          cloudTrainingParams={cloudTrainingParams}
          updateLocalTrainingParams={updateLocalTrainingParams}
          updateCloudTrainingParams={updateCloudTrainingParams}
          trainingType={trainingType}
          setTrainingType={setTrainingType}
          cloudTrainingStatus={cloudTrainingStatus}
          isPauseRequested={isPauseRequested}
          pauseStatus={pauseStatus}
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
      </div>
    </div>
  );
}
