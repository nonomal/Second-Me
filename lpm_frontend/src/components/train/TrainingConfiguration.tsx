'use client';

import type React from 'react';
import { useMemo, useState } from 'react';
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline';
import { Tabs, message, Spin } from 'antd';
import type { TabsProps } from 'antd';
import type { TrainingConfig } from '@/service/train';
import LocalTrainingConfig from './LocalTrainingConfig';
import CloudTrainingConfig from './CloudTrainingConfig';
import classNames from 'classnames';

interface BaseModelOption {
  value: string;
  label: string;
}

interface ModelConfig {
  provider_type?: string;
  [key: string]: any;
}

interface TrainingConfigurationProps {
  baseModelOptions: BaseModelOption[];
  modelConfig: ModelConfig | null;
  isTraining: boolean;
  updateTrainingParams: (params: TrainingConfig) => void;
  status: string;
  trainSuspended: boolean;
  handleResetProgress: () => void;
  handleTrainingAction: () => Promise<void>;
  trainActionLoading: boolean;
  setSelectedInfo: React.Dispatch<React.SetStateAction<boolean>>;
  trainingParams: TrainingConfig;
  cudaAvailable: boolean;
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
  cudaAvailable
}) => {
  const [activeTabKey, setActiveTabKey] = useState<string>('local');

  const disabledChangeParams = useMemo(() => {
    return isTraining || trainSuspended;
  }, [isTraining, trainSuspended]);

  const trainButtonText = useMemo(() => {
    return isTraining
      ? 'Stop Training'
      : status === 'trained'
        ? 'Retrain'
        : trainSuspended
          ? 'Resume Training'
          : 'Start Training';
  }, [isTraining, status, trainSuspended]);

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
  
  const tabItems: TabsProps['items'] = [
    {
      key: 'local',
      label: 'Local Training',
      children: (
        <LocalTrainingConfig
          baseModelOptions={baseModelOptions}
          modelConfig={modelConfig}
          isTraining={isTraining}
          updateTrainingParams={updateTrainingParams}
          status={status}
          trainSuspended={trainSuspended}
          trainingParams={trainingParams}
          cudaAvailable={cudaAvailable}
        />
      )
    },
    {
      key: 'cloud',
      label: 'Cloud Training',
      children: (
        <CloudTrainingConfig
          baseModelOptions={baseModelOptions}
          modelConfig={modelConfig}
          isTraining={isTraining}
          updateTrainingParams={updateTrainingParams}
          status={status}
          trainSuspended={trainSuspended}
          trainingParams={trainingParams}
          cudaAvailable={cudaAvailable}
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
        {`Configure how your Second Me will be trained using your memory data and identity. Then click 'Start Training'.`}
      </p>

      <Tabs activeKey={activeTabKey} onChange={setActiveTabKey} items={tabItems} className="mb-6" />

      <div className="flex justify-end items-center gap-4 pt-4 border-t mt-4">
        {isTraining && (
          <div className="flex items-center text-amber-600 bg-amber-50 px-3 py-2 rounded-md border border-amber-200">
            <StopIcon className="h-5 w-5 mr-2" />
            <span className="font-medium">Full stop only when the current step is complete</span>
          </div>
        )}

        {trainButtonText === 'Resume Training' && (
          <button
            className={`inline-flex items-center justify-center px-4 py-2 bg-red-600 hover:bg-red-700 border border-transparent text-sm font-medium rounded-md shadow-sm text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
            onClick={() => {
              if (!isTraining) {
                message.warning('Please do not shutdown your computer during training');
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
          onClick={() => {
            if (!isTraining) {
              message.warning('Please do not shutdown your computer during training');
            }

            handleTrainingAction();
          }}
        >
          {trainButtonIcon}
          {trainButtonText}
        </button>
      </div>
    </div>
  );
};

export default TrainingConfiguration;
