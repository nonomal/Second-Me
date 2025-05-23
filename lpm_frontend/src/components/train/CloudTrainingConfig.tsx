'use client';

import type React from 'react';
import { Fragment, useEffect, useMemo, useState } from 'react';
import { Listbox, Transition } from '@headlessui/react';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { Checkbox, InputNumber, message, Radio, Tooltip } from 'antd';
import type { TrainingConfig } from '@/service/train';
import { EVENT } from '../../utils/event';
import OpenAiModelIcon from '../svgs/OpenAiModelIcon';
import CustomModelIcon from '../svgs/CustomModelIcon';
import AlibabaCloudIcon from '../svgs/AlibabaCloudIcon';
import ColumnArrowIcon from '../svgs/ColumnArrowIcon';
import DoneIcon from '../svgs/DoneIcon';
import ThinkingModelModal from '../ThinkingModelModal';
import { useModelConfigStore } from '@/store/useModelConfigStore';
import { useCloudProviderStore } from '@/store/useCloudProviderStore';
import classNames from 'classnames';
import CloudProviderModal from '../modelConfigModal/CloudProviderModal';

interface BaseModelOption {
  value: string;
  label: string;
}

interface ModelConfig {
  provider_type?: string;
  [key: string]: any;
}

interface CloudTrainingConfigProps {
  baseModelOptions: BaseModelOption[];
  modelConfig: ModelConfig | null;
  isTraining: boolean;
  updateTrainingParams: (params: TrainingConfig) => void;
  status: string;
  trainSuspended: boolean;
  trainingParams: TrainingConfig;
  cudaAvailable: boolean;
}

const synthesisModeOptions = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' }
];

const CloudTrainingConfig: React.FC<CloudTrainingConfigProps> = ({
  baseModelOptions,
  modelConfig,
  isTraining,
  updateTrainingParams,
  trainingParams,
  status,
  trainSuspended,
  cudaAvailable
}) => {
  const [openThinkingModel, setOpenThinkingModel] = useState<boolean>(false);
  const [showThinkingWarning, setShowThinkingWarning] = useState<boolean>(false);
  const [openCloudProviderModal, setOpenCloudProviderModal] = useState<boolean>(false);
  const thinkingModelConfig = useModelConfigStore((state) => state.thinkingModelConfig);
  
  const cloudConfig = useCloudProviderStore((state) => state.cloudConfig);
  const updateCloudConfig = useCloudProviderStore((state) => state.updateCloudConfig);
  const saveCloudConfig = useCloudProviderStore((state) => state.saveCloudConfig);
  const fetchCloudConfig = useCloudProviderStore((state) => state.fetchCloudConfig);

  useEffect(() => {
    // 添加云提供商模态框事件监听
    const handleShowCloudProviderModal = () => {
      setOpenCloudProviderModal(true);
    };

    window.addEventListener(EVENT.SHOW_CLOUD_PROVIDER_MODAL, handleShowCloudProviderModal);

    // 初始加载云配置
    fetchCloudConfig();

    return () => {
      window.removeEventListener(EVENT.SHOW_CLOUD_PROVIDER_MODAL, handleShowCloudProviderModal);
    };
  }, [fetchCloudConfig]);

  const disabledChangeParams = useMemo(() => {
    return isTraining || trainSuspended;
  }, [isTraining, trainSuspended]);

  const thinkingConfigComplete = useMemo(() => {
    return (
      !!thinkingModelConfig.thinking_model_name &&
      !!thinkingModelConfig.thinking_api_key &&
      !!thinkingModelConfig.thinking_endpoint
    );
  }, [thinkingModelConfig]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-10">
        <div className="flex flex-col gap-2">
          <h4 className="text-base font-semibold text-gray-800 flex items-center">
            Step 1: Choose Cloud Provider
          </h4>
          {!cloudConfig.provider_type ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <label className="block text-sm font-medium text-red-500 mb-1">
                  None Cloud Provider Configured
                </label>
                <button
                  className="ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setOpenCloudProviderModal(true);
                  }}
                >
                  Configure Cloud Provider
                </button>
              </div>
              <span className="text-xs text-gray-500">
                Configure your cloud training with Alibaba Cloud Model Studio
              </span>
            </div>
          ) : (
            <div className="flex flex-col rounded-lg bg-white py-2  text-left">
              <div className="flex items-center">
                <span className="text-sm font-medium text-gray-700">Cloud Service Used : &nbsp;</span>
                <AlibabaCloudIcon className="h-5 w-5 mr-2 text-orange-600" />
                <span className="font-medium">
                    {cloudConfig.provider_type === 'alibaba' ? 'Alibaba Cloud Model Studio' : 'Custom Cloud'}
                </span>
                <button
                  className={classNames(
                    'ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10',
                    disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                  )}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    if (disabledChangeParams) {
                      message.warning('Cancel the current training run to configure the cloud provider');
                      return;
                    }

                    setOpenCloudProviderModal(true);
                  }}
                >
                  Configure Cloud Provider for Model Training
                </button>
              </div>
              <div className="mt-2 text-xs text-gray-500">
                This will provide faster training but requires API keys and may incur charges based on token usage
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <h4 className="text-base font-semibold text-gray-800 flex items-center">
            Step 2: Choose Support Model for Data Synthesis
          </h4>
          {!modelConfig?.provider_type ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <label className="block text-sm font-medium text-red-500 mb-1">
                  None Support Model for Data Synthesis
                </label>
                <button
                  className="ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    window.dispatchEvent(new CustomEvent(EVENT.SHOW_MODEL_CONFIG_MODAL));
                  }}
                >
                  Configure Support Model
                </button>
              </div>
              <span className="text-xs text-gray-500">
                Model used for processing and synthesizing your memory data
              </span>
            </div>
          ) : (
            <div className="flex items-center relative w-full rounded-lg bg-white py-2 text-left">
              <div className="flex items-center">
                <span className="text-sm font-medium text-gray-700">Model Used : &nbsp;</span>
                {modelConfig.provider_type === 'openai' ? (
                  <OpenAiModelIcon className="h-5 w-5 mr-2 text-green-600" />
                ) : (
                  <CustomModelIcon className="h-5 w-5 mr-2 text-blue-600" />
                )}
                <span className="font-medium">
                  {modelConfig.provider_type === 'openai' ? 'OpenAI' : 'Custom Model'}
                </span>
                <button
                  className={classNames(
                    'ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10',
                    disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                  )}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    if (disabledChangeParams) {
                      message.warning('Cancel the current training run to configure the model');
                      return;
                    }

                    window.dispatchEvent(new CustomEvent(EVENT.SHOW_MODEL_CONFIG_MODAL));
                  }}
                >
                  Configure Model for Data Synthesis
                </button>
              </div>
              <span className="ml-auto text-xs text-gray-500">
                Model used for processing and synthesizing your memory data
              </span>
            </div>
          )}
          <div className="flex flex-col gap-3">
            <div className="font-medium">Data Synthesis Mode</div>
            <Radio.Group
              disabled={disabledChangeParams}
              onChange={(e) =>
                updateTrainingParams({
                  ...trainingParams,
                  data_synthesis_mode: e.target.value
                })
              }
              optionType="button"
              options={synthesisModeOptions}
              value={trainingParams.data_synthesis_mode}
            />

            <span className="text-xs text-gray-500">
              Low: Fast data synthesis. Medium: Balanced synthesis and speed. High: Rich
              synthesis, slower speed.
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-semibold text-gray-800 mb-1">
              Step 3: Choose Base Model for Training Second Me
            </h4>
            <span className="text-xs text-gray-500">
              Base model for training your Second Me. Choose based on your available system
              resources.
            </span>
          </div>
          <Listbox
            disabled={disabledChangeParams}
            onChange={(value) => updateTrainingParams({ ...trainingParams, model_name: value })}
            value={trainingParams.model_name}
          >
            <div className="relative mt-1">
              <Listbox.Button
                className={classNames(
                  'relative w-full cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300',
                  disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                )}
              >
                <span className="block truncate">
                  {baseModelOptions.find((option) => option.value === trainingParams.model_name)
                    ?.label || 'Select a model...'}
                </span>
                <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                  <ColumnArrowIcon className="h-5 w-5 text-gray-400" />
                </span>
              </Listbox.Button>
              <Transition
                as={Fragment}
                leave="transition ease-in duration-100"
                leaveFrom="opacity-100"
                leaveTo="opacity-0"
              >
                <Listbox.Options className="absolute mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 z-[1] focus:outline-none">
                  {baseModelOptions.map((option) => (
                    <Listbox.Option
                      key={option.value}
                      className={({ active }) =>
                        `relative cursor-pointer select-none py-2 pl-10 pr-4 ${active ? 'bg-blue-100 text-blue-900' : 'text-gray-900'}`
                      }
                      value={option.value}
                    >
                      {({ selected }) => (
                        <>
                          <span
                            className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}
                          >
                            {option.label}
                          </span>
                          {selected ? (
                            <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-blue-600">
                              <DoneIcon className="h-5 w-5" />
                            </span>
                          ) : null}
                        </>
                      )}
                    </Listbox.Option>
                  ))}
                </Listbox.Options>
              </Transition>
            </div>
          </Listbox>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <h4 className="text-base font-semibold text-gray-800 mb-1">
              Step 4: Configure Advanced Training Parameters
            </h4>
            <div className="text-xs text-gray-500">
              Adjust these parameters to control training quality and performance. Recommended
              settings will ensure stable training.
            </div>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-2">
              <div className="flex gap-3 items-center">
                <div className="font-medium">Learning Rate</div>
                <Tooltip title="Lower values provide stable but slower learning, while higher values accelerate learning but risk overshooting optimal parameters, potentially causing training instability.">
                  <QuestionCircleOutlined className="cursor-pointer" />
                </Tooltip>
              </div>
              <InputNumber
                className="!w-[300px]"
                disabled={disabledChangeParams}
                max={0.005}
                min={0.00003}
                onChange={(value) => {
                  if (value == null) {
                    return;
                  }

                  updateTrainingParams({ ...trainingParams, learning_rate: value });
                }}
                status={
                  trainingParams.learning_rate == 0.005 || trainingParams.learning_rate == 0.00003
                    ? 'warning'
                    : undefined
                }
                step={0.0001}
                value={trainingParams.learning_rate}
              />
              <div className="text-xs text-gray-500">
                Enter a value between 0.00003 and 0.005 (recommended: 0.0001)
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex gap-3 items-center">
                <div className="font-medium">Number of Epochs</div>
                <Tooltip title="Controls how many complete passes the model makes through your entire dataset during training. More epochs allow deeper pattern recognition and memory integration but significantly increase training time and computational resources required.">
                  <QuestionCircleOutlined className="cursor-pointer" />
                </Tooltip>
              </div>
              <InputNumber
                className="!w-[300px]"
                disabled={disabledChangeParams}
                max={10}
                min={1}
                onChange={(value) => {
                  if (value == null) {
                    return;
                  }

                  updateTrainingParams({ ...trainingParams, number_of_epochs: value });
                }}
                status={
                  trainingParams.number_of_epochs == 10 || trainingParams.number_of_epochs == 1
                    ? 'warning'
                    : undefined
                }
                step={1}
                value={trainingParams.number_of_epochs}
              />
              <div className="text-xs text-gray-500">
                Enter an integer between 1 and 10 (recommended: 3)
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex gap-3 items-center">
                <div className="font-medium">Concurrency Threads</div>
                <Tooltip title="Defines the number of parallel processing streams used during data synthesis. Higher values can reduce overall training time but increase system resource consumption and may trigger API rate limits, potentially causing training failures.">
                  <QuestionCircleOutlined className="cursor-pointer" />
                </Tooltip>
              </div>
              <InputNumber
                className="!w-[300px]"
                disabled={disabledChangeParams}
                max={10}
                min={1}
                onChange={(value) => {
                  if (value == null) {
                    return;
                  }

                  updateTrainingParams({ ...trainingParams, concurrency_threads: value });
                }}
                status={
                  trainingParams.concurrency_threads == 10 ||
                  trainingParams.concurrency_threads == 1
                    ? 'warning'
                    : undefined
                }
                step={1}
                value={trainingParams.concurrency_threads}
              />
              <div className="text-xs text-gray-500">
                Enter an integer between 1 and 10 (recommended: 2)
              </div>
            </div>
          </div>
        </div>
      </div>

      <ThinkingModelModal onClose={() => setOpenThinkingModel(false)} open={openThinkingModel} />
      <CloudProviderModal 
        onClose={() => setOpenCloudProviderModal(false)} 
        open={openCloudProviderModal} 
        cloudConfig={cloudConfig}
        updateCloudConfig={updateCloudConfig}
        saveCloudConfig={saveCloudConfig}
      />
    </div>
  );
};

export default CloudTrainingConfig;
