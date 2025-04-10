'use client';

import type React from 'react';
import { Fragment, useEffect, useState } from 'react';
import { Listbox, Transition } from '@headlessui/react';
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline';
import { EVENT } from '@/utils/event';
import { InputNumber, Radio, Spin, Tooltip } from 'antd';
import type { TrainingParams } from '@/service/train';
import { QuestionCircleOutlined } from '@ant-design/icons';

interface BaseModelOption {
  value: string;
  label: string;
}

interface TrainingConfig {
  modelProvider: string;
  baseModel: string;
  modelType: string;
  epochs: number;
  learningRate: string;
  memoryPriority: string;
  showAdvanced: boolean;
}

interface ModelConfig {
  provider_type?: string;
  [key: string]: any;
}

interface TrainingConfigurationProps {
  config: TrainingConfig;
  setConfig: React.Dispatch<React.SetStateAction<TrainingConfig>>;
  baseModelOptions: BaseModelOption[];
  modelConfig: ModelConfig | null;
  isTraining: boolean;
  updateTrainingParams: (params: TrainingParams) => void;
  status: string;
  isResume: boolean;
  nowTrainingParams: TrainingParams | null;
  changeBaseModel: boolean;
  handleTrainingAction: () => Promise<void>;
  trainActionLoading: boolean;
  setSelectedInfo: React.Dispatch<React.SetStateAction<boolean>>;
  trainingParams: TrainingParams;
}

const synthesisModeOptions = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' }
];

const TrainingConfiguration: React.FC<TrainingConfigurationProps> = ({
  config,
  setConfig,
  baseModelOptions,
  modelConfig,
  isTraining,
  updateTrainingParams,
  trainingParams,
  nowTrainingParams,
  status,
  isResume,
  changeBaseModel,
  trainActionLoading,
  handleTrainingAction,
  setSelectedInfo
}) => {
  const [disabledChangeParams, setDisabledChangeParams] = useState<boolean>(false);

  useEffect(() => {
    setDisabledChangeParams(isTraining || (isResume && !changeBaseModel));
  }, [isTraining, isResume, changeBaseModel]);

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

      <div className="space-y-6">
        <div className="space-y-8">
          <div>
            <h4 className="text-base font-semibold text-gray-800 mb-4 flex items-center">
              Step 1: Choose Support Model for Data Synthesis
            </h4>
            <div className="space-y-4">
              <div>
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
                    <div
                      className="flex items-center cursor-pointer"
                      onClick={() => {
                        window.dispatchEvent(new CustomEvent(EVENT.SHOW_MODEL_CONFIG_MODAL));
                      }}
                    >
                      <span className="text-sm font-medium text-gray-700">Model Used : &nbsp;</span>
                      {modelConfig.provider_type === 'openai' ? (
                        <svg
                          className="h-5 w-5 mr-2 text-green-600"
                          fill="currentColor"
                          viewBox="0 0 24 24"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" />
                        </svg>
                      ) : (
                        <svg
                          className="h-5 w-5 mr-2 text-blue-600"
                          fill="none"
                          stroke="currentColor"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          viewBox="0 0 24 24"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                          <polyline points="7.5 4.21 12 6.81 16.5 4.21" />
                          <polyline points="7.5 19.79 7.5 14.6 3 12" />
                          <polyline points="21 12 16.5 14.6 16.5 19.79" />
                          <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                          <line x1="12" x2="12" y1="22.08" y2="12" />
                        </svg>
                      )}
                      <span className="font-medium">
                        {modelConfig.provider_type === 'openai' ? 'OpenAI' : 'Custom Model'}
                      </span>
                      <button
                        className="ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
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
                    value={
                      disabledChangeParams && nowTrainingParams && !changeBaseModel
                        ? nowTrainingParams.data_synthesis_mode
                        : trainingParams.data_synthesis_mode
                    }
                  />

                  <span className="text-xs text-gray-500">
                    Low: Fast data synthesis. Medium: Balanced synthesis and speed. High: Rich
                    synthesis, slower speed.
                  </span>
                </div>
              </div>

              <div className="mt-8">
                <div className="flex items-center justify-between">
                  <h4 className="text-base font-semibold text-gray-800 mb-1">
                    Step 2: Choose Base Model for Training Second Me
                  </h4>
                  <span className="text-xs text-gray-500">
                    Base model for training your Second Me. Choose based on your available system
                    resources.
                  </span>
                </div>
                <Listbox
                  disabled={isTraining || trainActionLoading}
                  onChange={(value) => setConfig({ ...config, baseModel: value })}
                  value={config.baseModel}
                >
                  <div className="relative mt-1">
                    <Listbox.Button className="relative w-full cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300">
                      <span className="block truncate">
                        {baseModelOptions.find((option) => option.value === config.baseModel)
                          ?.label || 'Select a model...'}
                      </span>
                      <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                        <svg
                          className="h-5 w-5 text-gray-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            d="M7 7l3-3 3 3m0 6l-3 3-3-3"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="1.5"
                          />
                        </svg>
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
                                    <svg
                                      className="h-5 w-5"
                                      fill="currentColor"
                                      viewBox="0 0 20 20"
                                    >
                                      <path
                                        clipRule="evenodd"
                                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                        fillRule="evenodd"
                                      />
                                    </svg>
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

              <div className="mt-8">
                <div className="flex justify-between items-center">
                  <h4 className="text-base font-semibold text-gray-800 mb-1">
                    Step 3: Configure Advanced Training Parameters
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
                      <Tooltip title="Lower values (0.0001) provide stable but slower learning, while higher values accelerate learning but risk overshooting optimal parameters, potentially causing training instability.">
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
                        trainingParams.learning_rate == 0.005 ||
                        trainingParams.learning_rate == 0.00003
                          ? 'warning'
                          : undefined
                      }
                      step={0.00001}
                      value={
                        disabledChangeParams && nowTrainingParams && !changeBaseModel
                          ? nowTrainingParams.learning_rate
                          : trainingParams.learning_rate
                      }
                    />
                    {trainingParams.learning_rate == 0.00003 ||
                    trainingParams.learning_rate == 0.005 ? (
                      <div className="text-xs text-red-500">
                        Please enter a valid number between 0.00003 and 0.005
                      </div>
                    ) : (
                      <div className="text-xs text-gray-500">
                        Controls how quickly the model adapts to new information during training.
                      </div>
                    )}
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
                        trainingParams.number_of_epochs == 10 ||
                        trainingParams.number_of_epochs == 1
                          ? 'warning'
                          : undefined
                      }
                      step={1}
                      value={
                        disabledChangeParams && nowTrainingParams && !changeBaseModel
                          ? nowTrainingParams.number_of_epochs
                          : trainingParams.number_of_epochs
                      }
                    />
                    {trainingParams.number_of_epochs == 10 ||
                    trainingParams.number_of_epochs == 1 ? (
                      <div className="text-xs text-red-500">
                        Please enter a whole number between 1 and 10
                      </div>
                    ) : (
                      <div className="text-xs text-gray-500">
                        Higher values result in deeper memory but longer training time.
                      </div>
                    )}
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
                      value={
                        disabledChangeParams && nowTrainingParams && !changeBaseModel
                          ? nowTrainingParams.concurrency_threads
                          : trainingParams.concurrency_threads
                      }
                    />
                    {trainingParams.concurrency_threads == 10 ||
                    trainingParams.concurrency_threads == 1 ? (
                      <div className="text-xs text-red-500">
                        Please enter a whole number between 1 and 10
                      </div>
                    ) : (
                      <div className="text-xs text-gray-500">
                        Higher values may cause system errors or API request failures.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {config.showAdvanced && (
          <div className="space-y-4 border-t pt-4 mt-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Base Model</label>
              <Listbox disabled={true} onChange={() => {}} value="Qwen 7B">
                <div className="relative mt-1">
                  <Listbox.Button className="relative w-full cursor-not-allowed rounded-lg bg-gray-50 py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300">
                    <span className="block truncate">Qwen 7B</span>
                    <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                      <svg
                        className="h-5 w-5 text-gray-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          d="M7 7l3-3 3 3m0 6l-3 3-3-3"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="1.5"
                        />
                      </svg>
                    </span>
                  </Listbox.Button>
                </div>
              </Listbox>
            </div>
          </div>
        )}

        <div className="flex justify-end items-center gap-4 pt-4 border-t mt-4">
          {isTraining && (
            <div className="flex items-center text-amber-600 bg-amber-50 px-3 py-2 rounded-md border border-amber-200">
              <svg
                className="h-5 w-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                />
              </svg>
              <span className="font-medium">Full stop only when the current step is complete</span>
            </div>
          )}
          <button
            className={`inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${isTraining ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'}
            ${!isTraining && !modelConfig?.provider_type ? 'bg-gray-300 hover:bg-gray-400 cursor-not-allowed' : 'cursor-pointer'}
            focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
            disabled={!isTraining && !modelConfig?.provider_type}
            onClick={handleTrainingAction}
          >
            {isTraining ? (
              <>
                {trainActionLoading ? (
                  <Spin className="h-5 w-5 mr-2" />
                ) : (
                  <StopIcon className="h-5 w-5 mr-2" />
                )}
                Stop Training
              </>
            ) : (
              <>
                <PlayIcon className="h-5 w-5 mr-2" />
                {!changeBaseModel
                  ? status === 'trained' || status === 'running'
                    ? 'Retrain'
                    : isResume
                      ? 'Resume Training'
                      : 'Start Training'
                  : 'Start Training'}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default TrainingConfiguration;
