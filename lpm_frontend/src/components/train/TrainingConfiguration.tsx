'use client';

import type React from 'react';
import { useMemo } from 'react';
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline';
import { Tabs, message, Spin, Tooltip } from 'antd';
import type { TabsProps } from 'antd';
import type { LocalTrainingParams, CloudTrainingParams } from '@/service/train';
import type { IModelConfig } from '@/service/modelConfig';
import LocalTrainingConfig from './LocalTrainingConfig';
import CloudTrainingConfig from './CloudTrainingConfig';

interface BaseModelOption {
  value: string;
  label: string;
}

interface TrainingConfigurationProps {
  baseModelOptions: BaseModelOption[];
  modelConfig: IModelConfig | null;
  isTraining: boolean;
  updateLocalTrainingParams: (params: Partial<LocalTrainingParams>) => void;
  updateCloudTrainingParams: (params: Partial<CloudTrainingParams>) => void;
  status: string;
  trainSuspended: boolean;
  handleResetProgress: () => void;
  handleTrainingAction: (trainingType: 'local' | 'cloud') => Promise<void>;
  trainActionLoading: boolean;
  setSelectedInfo: React.Dispatch<React.SetStateAction<boolean>>;
  localTrainingParams: LocalTrainingParams;
  cloudTrainingParams: CloudTrainingParams;
  cudaAvailable: boolean;
  trainingType: 'local' | 'cloud';
  setTrainingType: (type: 'local' | 'cloud') => void;
  cloudTrainingStatus?: 'idle' | 'training' | 'trained' | 'failed' | 'suspended';
  isPauseRequested?: boolean;
  pauseStatus?: 'success' | 'pending' | 'failed' | null;
}

const TrainingConfiguration: React.FC<TrainingConfigurationProps> = ({
  baseModelOptions,
  modelConfig,
  isTraining,
  updateLocalTrainingParams,
  updateCloudTrainingParams,
  localTrainingParams,
  cloudTrainingParams,
  status,
  handleResetProgress,
  trainSuspended,
  trainActionLoading,
  handleTrainingAction,
  setSelectedInfo,
  cudaAvailable,
  trainingType,
  setTrainingType,
  cloudTrainingStatus = 'idle',
  isPauseRequested = false,
  pauseStatus = null
}) => {
  const activeTabKey = trainingType;

  const trainButtonText = useMemo(() => {
    if (isTraining) {
      // Show specific message when pausing is pending
      if (activeTabKey === 'cloud' && isPauseRequested && pauseStatus === 'pending') {
        return 'Stopping...';
      }

      return 'Stop Training';
    }
    
    if (activeTabKey === 'cloud') {
      // Cloud training button logic
      if (cloudTrainingStatus === 'trained') {
        return 'Retrain on Cloud';
      }

      if (cloudTrainingStatus === 'suspended' || trainSuspended) {
        return 'Resume Training';
      }

      if (cloudTrainingStatus === 'failed') {
        return 'Retry Cloud Training';
      }

      return 'Start Cloud Training';
    }
    
    // Local training button logic
    if (status === 'trained') {
      return 'Retrain Locally';
    }

    if (trainSuspended) {
      return 'Resume Training';
    }

    return 'Start Local Training';
  }, [
    isTraining,
    status,
    trainSuspended,
    activeTabKey,
    cloudTrainingStatus,
    isPauseRequested,
    pauseStatus
  ]);

  const trainButtonIcon = useMemo(() => {
    return isTraining ? (
      trainActionLoading ||
      (activeTabKey === 'cloud' && isPauseRequested && pauseStatus === 'pending') ? (
        <Spin className="h-5 w-5 mr-2" />
      ) : (
        <StopIcon className="h-5 w-5 mr-2" />
      )
    ) : (
      <PlayIcon className="h-5 w-5 mr-2" />
    );
  }, [isTraining, trainActionLoading, activeTabKey, isPauseRequested, pauseStatus]);

  const handleTraining = async () => {
    if (!isTraining && !trainSuspended) {
      if (activeTabKey === 'cloud') {
        message.warning('Please ensure stable internet connection during cloud training');
      } else {
        message.warning('Please do not shutdown your computer during training');
      }
    }

    if (activeTabKey === 'local') {
      await handleTrainingAction('local');
    } else {
      if (!modelConfig?.cloud_service_api_key) {
        message.error('Please set up cloud service API key first');

        return;
      }

      if (isTraining) {
        await handleTrainingAction('cloud');

        return;
      }

      await handleTrainingAction('cloud');
    }
  };

  // Handle tab change with training status check
  const handleTabChange = (key: string) => {
    // Check if any training is active (running or suspended)
    const isLocalTrainingActive = trainingType === 'local' && (isTraining || trainSuspended);
    const isCloudTrainingActive =
      trainingType === 'cloud' && (isTraining || cloudTrainingStatus === 'suspended');
    
    // If any training is active, prevent switching and show warning
    if (isLocalTrainingActive || isCloudTrainingActive) {
      if (key === 'local' && trainingType === 'cloud') {
        if (isTraining) {
          message.warning(
            'Cannot switch to local training while cloud training is in progress. Please stop cloud training first.'
          );
        } else {
          message.warning(
            'Cannot switch to local training while cloud training is suspended. Please reset or complete cloud training first.'
          );
        }

        return;
      }

      if (key === 'cloud' && trainingType === 'local') {
        if (isTraining) {
          message.warning(
            'Cannot switch to cloud training while local training is in progress. Please stop local training first.'
          );
        } else {
          message.warning(
            'Cannot switch to cloud training while local training is suspended. Please reset or complete local training first.'
          );
        }

        return;
      }
    }

    // Allow tab change if no training is active or switching to the same type
    setTrainingType(key as 'local' | 'cloud');
  };

  // Helper function to get tooltip message for disabled tabs
  const getDisabledTooltip = (tabType: 'local' | 'cloud'): string => {
    if (tabType === 'local' && trainingType === 'cloud') {
      if (isTraining) {
        return 'Cannot switch to local training while cloud training is in progress. Please stop cloud training first.';
      }

      if (cloudTrainingStatus === 'suspended') {
        return 'Cannot switch to local training while cloud training is suspended. Please reset or complete cloud training first.';
      }
    }
    
    if (tabType === 'cloud' && trainingType === 'local') {
      if (isTraining) {
        return 'Cannot switch to cloud training while local training is in progress. Please stop local training first.';
      }

      if (trainSuspended) {
        return 'Cannot switch to cloud training while local training is suspended. Please reset or complete local training first.';
      }
    }
    
    return '';
  };

  const tabItems: TabsProps['items'] = [
    {
      key: 'local',
      label: (
        <Tooltip
          placement="top"
          title={
            (isTraining && trainingType === 'cloud') ||
            (trainingType === 'cloud' && cloudTrainingStatus === 'suspended')
              ? getDisabledTooltip('local')
              : ''
          }
        >
          <span
            className={
              (isTraining && trainingType === 'cloud') ||
              (trainingType === 'cloud' && cloudTrainingStatus === 'suspended')
                ? 'text-gray-400 cursor-not-allowed'
                : ''
            }
          >
            Local Training
          </span>
        </Tooltip>
      ),
      disabled:
        (isTraining && trainingType === 'cloud') ||
        (trainingType === 'cloud' && cloudTrainingStatus === 'suspended'),
      children: (
        <LocalTrainingConfig
          baseModelOptions={baseModelOptions}
          cudaAvailable={cudaAvailable}
          isTraining={isTraining}
          modelConfig={modelConfig}
          status={status}
          trainSuspended={trainSuspended}
          trainingParams={localTrainingParams}
          updateTrainingParams={updateLocalTrainingParams}
        />
      )
    },
    {
      key: 'cloud',
      label: (
        <Tooltip
          placement="top"
          title={
            (isTraining && trainingType === 'local') || (trainingType === 'local' && trainSuspended)
              ? getDisabledTooltip('cloud')
              : ''
          }
        >
          <span
            className={
              (isTraining && trainingType === 'local') ||
              (trainingType === 'local' && trainSuspended)
                ? 'text-gray-400 cursor-not-allowed'
                : ''
            }
          >
            Cloud Training
          </span>
        </Tooltip>
      ),
      disabled:
        (isTraining && trainingType === 'local') || (trainingType === 'local' && trainSuspended),
      children: (
        <CloudTrainingConfig
          cudaAvailable={cudaAvailable}
          isTraining={isTraining}
          modelConfig={modelConfig as IModelConfig | null}
          status={status}
          trainSuspended={trainSuspended}
          trainingParams={cloudTrainingParams}
          updateTrainingParams={updateCloudTrainingParams}
        />
      )
    }
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold tracking-tight text-gray-900">
          Training Configuration
        </h2>
        <button
          className="p-1.5 rounded-full bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700 transition-colors"
          onClick={() => setSelectedInfo(true)}
          title="Learn more about training process"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
            />
          </svg>
        </button>
      </div>
      <p className="text-gray-600 mb-6 leading-relaxed">
        {`Configure how your Second Me will be trained using your memory data and identity. Then click '${activeTabKey === 'local' ? 'Start Local Training' : 'Start Cloud Training'}'.`}
      </p>

      <Tabs activeKey={activeTabKey} className="mb-6" items={tabItems} onChange={handleTabChange} />

      <div className="flex justify-end items-center gap-4 pt-4 border-t mt-4">
        {isTraining && (
          <div className="flex items-center text-amber-600 bg-amber-50 px-3 py-2 rounded-md border border-amber-200">
            <StopIcon className="h-5 w-5 mr-2" />
            <span className="font-medium">Full stop only when the current step is complete</span>
          </div>
        )}

        {((activeTabKey === 'local' && trainSuspended) ||
          (activeTabKey === 'cloud' &&
            (cloudTrainingStatus === 'suspended' || cloudTrainingStatus === 'failed'))) && (
          <button
            className={`inline-flex items-center justify-center px-4 py-2 bg-red-600 hover:bg-red-700 border border-transparent text-sm font-medium rounded-md shadow-sm text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
            onClick={() => {
              if (!isTraining && !trainSuspended) {
                if (activeTabKey === 'cloud') {
                  message.warning('Please ensure stable internet connection during cloud training');
                } else {
                  message.warning('Please do not shutdown your computer during training');
                }
              }

              handleResetProgress();
            }}
          >
            <StopIcon className="h-5 w-5 mr-2" />
            Reset Training
          </button>
        )}
        <button
          className={`inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${isTraining ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'}
          ${!isTraining && !modelConfig?.provider_type ? 'bg-gray-300 hover:bg-gray-400 cursor-not-allowed' : 'cursor-pointer'}
          focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
          disabled={!isTraining && !modelConfig?.provider_type}
          onClick={handleTraining}
        >
          {trainButtonIcon}
          {trainButtonText}
        </button>
      </div>
    </div>
  );
};

export default TrainingConfiguration;
