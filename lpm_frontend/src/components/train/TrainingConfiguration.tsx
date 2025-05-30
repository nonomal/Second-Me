'use client';

import type React from 'react';
import { useMemo } from 'react';
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline';
import { Tabs, message, Spin } from 'antd';
import type { TabsProps } from 'antd';
import type { TrainingConfig } from '@/service/train';
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
  updateTrainingParams: (params: TrainingConfig) => void;
  status: string;
  trainSuspended: boolean;
  handleResetProgress: () => void;
  handleTrainingAction: (trainingType: 'local' | 'cloud') => Promise<void>;
  trainActionLoading: boolean;
  setSelectedInfo: React.Dispatch<React.SetStateAction<boolean>>;
  trainingParams: TrainingConfig;
  cudaAvailable: boolean;
  trainingType: 'local' | 'cloud';
  setTrainingType: (type: 'local' | 'cloud') => void;
  cloudTrainingStatus?: 'idle' | 'training' | 'trained' | 'failed' | 'suspended';
}

const TrainingConfiguration: React.FC<TrainingConfigurationProps> = ({
  baseModelOptions,
  modelConfig,
  isTraining,
  updateTrainingParams,
  trainingParams,
  status,
  handleResetProgress,
  trainSuspended,
  trainActionLoading,
  handleTrainingAction,
  setSelectedInfo,
  cudaAvailable,
  trainingType,
  setTrainingType,
  cloudTrainingStatus = 'idle'
}) => {
  // 使用从父组件传递下来的 trainingType 状态
  const activeTabKey = trainingType;

  const trainButtonText = useMemo(() => {
    if (isTraining) {
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
  }, [isTraining, status, trainSuspended, activeTabKey, cloudTrainingStatus]);

  const trainButtonIcon = useMemo(() => {
    return isTraining ? (
      trainActionLoading ? (
        <Spin className="h-5 w-5 mr-2" />
      ) : (
        <StopIcon className="h-5 w-5 mr-2" />
      )
    ) : (
      <PlayIcon className="h-5 w-5 mr-2" />
    );
  }, [isTraining, trainActionLoading]);
  
  // 处理不同模式的训练动作
  const handleTraining = async () => {
    if (!isTraining && !trainSuspended) {
      if (activeTabKey === 'cloud') {
        message.warning('Please ensure stable internet connection during cloud training');
      } else {
        message.warning('Please do not shutdown your computer during training');
      }
    }

    if (activeTabKey === 'local') {
      // 调用本地训练功能
      await handleTrainingAction('local');
    } else {
      // 检查是否设置了云服务 API Key
      if (!modelConfig?.cloud_service_api_key) {
        message.error('Please set up cloud service API key first');

        return;
      }

      if (isTraining) {
        // 如果正在训练，停止训练（无论是本地还是云端）
        await handleTrainingAction('cloud');

        return;
      }

      // 直接调用云端训练方法
      await handleTrainingAction('cloud');
    }
  };
  
  const tabItems: TabsProps['items'] = [
    {
      key: 'local',
      label: 'Local Training',
      children: (
        <LocalTrainingConfig
          baseModelOptions={baseModelOptions}
          cudaAvailable={cudaAvailable}
          isTraining={isTraining}
          modelConfig={modelConfig}
          status={status}
          trainSuspended={trainSuspended}
          trainingParams={trainingParams}
          updateTrainingParams={updateTrainingParams}
        />
      )
    },
    {
      key: 'cloud',
      label: 'Cloud Training',
      children: (
        <CloudTrainingConfig
          cudaAvailable={cudaAvailable}
          isTraining={isTraining}
          modelConfig={modelConfig as IModelConfig | null}
          status={status}
          trainSuspended={trainSuspended}
          trainingParams={trainingParams}
          updateTrainingParams={updateTrainingParams}
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

      <Tabs
        activeKey={activeTabKey}
        className="mb-6"
        items={tabItems}
        onChange={(key) => {
          // 首先设置活动标签，这样UI立即响应
          setTrainingType(key as 'local' | 'cloud');

          // 当切换标签时，切换到对应环境的模型，但只在必要时更新
          if (key === 'local') {
            // 从云端切换到本地，使用 local_model_name（如果存在）
            if (
              trainingParams.local_model_name &&
              trainingParams.model_name !== trainingParams.local_model_name
            ) {
              updateTrainingParams({
                ...trainingParams,
                model_name: trainingParams.local_model_name
              });
            }
          } else if (key === 'cloud') {
            // 从本地切换到云端，优先使用 cloud_model_name（如果存在）
            // 如果 cloud_model_name 存在，并且与 model_name 不同，则使用 cloud_model_name
            if (
              trainingParams.cloud_model_name &&
              trainingParams.model_name !== trainingParams.cloud_model_name
            ) {
              updateTrainingParams({
                ...trainingParams,
                model_name: trainingParams.cloud_model_name
              });
            }
          }
        }}
      />

      <div className="flex justify-end items-center gap-4 pt-4 border-t mt-4">
        {isTraining && (
          <div className="flex items-center text-amber-600 bg-amber-50 px-3 py-2 rounded-md border border-amber-200">
            <StopIcon className="h-5 w-5 mr-2" />
            <span className="font-medium">Full stop only when the current step is complete</span>
          </div>
        )}

        {((activeTabKey === 'local' && trainSuspended) || 
          (activeTabKey === 'cloud' && (cloudTrainingStatus === 'suspended' || cloudTrainingStatus === 'failed'))) && (
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
