'use client';

import type React from 'react';
import { Fragment, useEffect, useMemo, useState } from 'react';
import { Listbox, Transition } from '@headlessui/react'; // Removed HeadlessRadioGroup
import { QuestionCircleOutlined } from '@ant-design/icons';
import { InputNumber, message, Tooltip, Spin, Radio } from 'antd';
import type { RadioChangeEvent } from 'antd'; // Added RadioChangeEvent for typing
import type { TrainingConfig } from '@/service/train';
import { EVENT } from '../../utils/event';
import OpenAiModelIcon from '../svgs/OpenAiModelIcon';
import CustomModelIcon from '../svgs/CustomModelIcon';
import AlibabaCloudIcon from '../svgs/AlibabaCloudIcon';
import ColumnArrowIcon from '../svgs/ColumnArrowIcon';
import DoneIcon from '../svgs/DoneIcon';
import ThinkingModelModal from '../ThinkingModelModal';
import { useCloudProviderStore } from '@/store/useCloudProviderStore';
import classNames from 'classnames';
import CloudProviderModal from '../modelConfigModal/CloudProviderModal';
import { listAvailableModels } from '@/service/cloudService';
import type { CloudModel } from '@/service/cloudService';
import type { IModelConfig } from '@/service/modelConfig';

interface CloudTrainingConfigProps {
  modelConfig: IModelConfig | null;
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
  modelConfig,
  isTraining,
  updateTrainingParams,
  trainingParams,
  trainSuspended,
}) => {
  const [openThinkingModel, setOpenThinkingModel] = useState<boolean>(false);
  const [openCloudProviderModal, setOpenCloudProviderModal] = useState<boolean>(false);
  const [availableCloudModels, setAvailableCloudModels] = useState<CloudModel[]>([]);
  const [loadingModels, setLoadingModels] = useState<boolean>(false);
  const cloudConfig = useCloudProviderStore((state) => state.cloudConfig);
  const updateCloudConfig = useCloudProviderStore((state) => state.updateCloudConfig);
  const fetchCloudConfig = useCloudProviderStore((state) => state.fetchCloudConfig);

  useEffect(() => {
    const handleShowCloudProviderModal = () => {
      setOpenCloudProviderModal(true);
    };

    window.addEventListener(EVENT.SHOW_CLOUD_PROVIDER_MODAL, handleShowCloudProviderModal);
    fetchCloudConfig();

    return () => {
      window.removeEventListener(EVENT.SHOW_CLOUD_PROVIDER_MODAL, handleShowCloudProviderModal);
    };
  }, [fetchCloudConfig]);

  const disabledChangeParams = useMemo(() => {
    return isTraining || trainSuspended;
  }, [isTraining, trainSuspended]);

  const hasCloudServiceApiKey = useMemo(() => {
    return !!(modelConfig && modelConfig.cloud_service_api_key);
  }, [modelConfig]);

  useEffect(() => {
    // This effect seems to manage cloudConfig.provider_type based on modelConfig
    // It might be simplified or re-evaluated based on overall logic for provider_type
    if (hasCloudServiceApiKey && cloudConfig.provider_type !== 'alibaba') {
      updateCloudConfig({
        ...cloudConfig,
        provider_type: 'alibaba', // Assuming 'alibaba' is the only cloud provider for now
        cloud_service_api_key: modelConfig?.cloud_service_api_key || ''
      });
    } else if (!hasCloudServiceApiKey && cloudConfig.provider_type !== '') {
      updateCloudConfig({
        ...cloudConfig,
        provider_type: '',
        cloud_service_api_key: ''
      });
    }
  }, [hasCloudServiceApiKey, modelConfig, cloudConfig, updateCloudConfig]);

  useEffect(() => {
    const fetchModels = async () => {
      if (hasCloudServiceApiKey) {
        setLoadingModels(true);
        try {
          const res = await listAvailableModels();
          if (res.data.code === 0 && Array.isArray(res.data.data)) {
            const fetchedModels: CloudModel[] = res.data.data.map((item: any) => ({
              model_id: item.id,
              model_name: item.id, 
            }));
            setAvailableCloudModels(fetchedModels);

            if (fetchedModels.length > 0) {
              // 首先检查是否有 cloud_model_name 并且它在可用模型中
              let validCloudModelName = false;
              if (trainingParams.cloud_model_name) {
                validCloudModelName = fetchedModels.some(m => m.model_id === trainingParams.cloud_model_name);
              }

              // 然后检查 model_name 是否在可用模型中
              const currentModelIsValid = fetchedModels.some(m => m.model_id === trainingParams.model_name);
              
              // 决定要使用的模型ID
              let modelToUse = '';
              
              if (validCloudModelName) {
                // 如果有有效的 cloud_model_name，优先使用它
                modelToUse = trainingParams.cloud_model_name as string;
              } else if (currentModelIsValid) {
                // 其次使用当前有效的 model_name
                modelToUse = trainingParams.model_name;
              } else {
                // 最后使用第一个可用模型
                modelToUse = fetchedModels[0].model_id;
              }
              
              // 只有在需要更新时才更新状态
              if (trainingParams.model_name !== modelToUse || trainingParams.cloud_model_name !== modelToUse) {
                updateTrainingParams({ 
                  ...trainingParams, 
                  model_name: modelToUse,
                  cloud_model_name: modelToUse 
                });
              }
            } else {
              // 如果没有获取到模型，且当前有模型选择，则清空选择
              if (trainingParams.model_name !== '' || trainingParams.cloud_model_name !== '') {
                updateTrainingParams({ 
                  ...trainingParams, 
                  model_name: '',
                  cloud_model_name: '' 
                });
              }
            }
          } else {
            message.error(res.data.message || 'Failed to fetch available cloud models');
            setAvailableCloudModels([]);
            // 只有在当前有模型选择的情况下才清空
            if (trainingParams.model_name !== '' || trainingParams.cloud_model_name !== '') {
              updateTrainingParams({ 
                ...trainingParams, 
                model_name: '',
                cloud_model_name: '' 
              });
            }
          }
        } catch (error: any) { // Added type for error
          message.error(error.message || 'Error fetching cloud models');
          setAvailableCloudModels([]);

          // 只有在当前有模型选择的情况下才清空
          if (trainingParams.model_name !== '' || trainingParams.cloud_model_name !== '') {
             updateTrainingParams({ 
               ...trainingParams, 
               model_name: '',
               cloud_model_name: '' 
             });
          }

          console.error('Failed to fetch cloud models:', error);
        }
        setLoadingModels(false);
      } else {
        setAvailableCloudModels([]);
        // 只有在当前有模型选择的情况下才清空
        if (trainingParams.model_name !== '' || trainingParams.cloud_model_name !== '') {
          updateTrainingParams({ 
            ...trainingParams, 
            model_name: '',
            cloud_model_name: '' 
          });
        }
      }
    };

    fetchModels();
  }, [hasCloudServiceApiKey]); // 移除 trainingParams.model_name 依赖，避免循环请求

  // 移除不必要的同步逻辑，避免循环更新
  // useEffect(() => {
  //   // 确保云训练模型和当前模型同步
  //   if (trainingParams.cloud_model_name && trainingParams.model_name !== trainingParams.cloud_model_name) {
  //     updateTrainingParams({
  //       ...trainingParams,
  //       model_name: trainingParams.cloud_model_name
  //     });
  //   }
  // }, [trainingParams, updateTrainingParams]);

  const cloudModelOptions = useMemo(() => {
    return availableCloudModels.map((model) => ({
      value: model.model_id,
      label: model.model_name
    }));
  }, [availableCloudModels]);

  const handleSynthesisModeChange = (e: RadioChangeEvent) => {
    // Added type for event
    updateTrainingParams({
      ...trainingParams,
      data_synthesis_mode: e.target.value
    });
  };



  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-10">
        {/* Step 1: Choose Cloud Provider */}
        <div className="flex flex-col gap-2">
          <h4 className="text-base font-semibold text-gray-800 flex items-center">
            Step 1: Choose Cloud Provider
          </h4>
          {!hasCloudServiceApiKey ? (
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
            <div className="flex flex-col rounded-lg bg-white py-2 text-left">
              <div className="flex items-center">
                <span className="text-sm font-medium text-gray-700">
                  Cloud Service Used : &nbsp;
                </span>
                <AlibabaCloudIcon className="h-5 w-5 mr-2 text-orange-600" />
                <span className="font-medium">
                  {cloudConfig.provider_type === 'alibaba'
                    ? 'Alibaba Cloud Model Studio'
                    : 'Custom Cloud'}
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
                      message.warning(
                        'Cancel the current training run to configure the cloud provider'
                      );

                      return;
                    }

                    setOpenCloudProviderModal(true);
                  }}
                >
                  Configure Cloud Provider for Model Training
                </button>
              </div>
              <div className="mt-2 text-xs text-gray-500">
                This will provide faster training but requires API keys and may incur charges based
                on token usage
              </div>
            </div>
          )}
        </div>

        {/* Step 2: Choose Support Model for Data Synthesis */}
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
              onChange={handleSynthesisModeChange} // Use typed handler
              optionType="button"
              options={synthesisModeOptions}
              value={trainingParams.data_synthesis_mode}
            />
            <span className="text-xs text-gray-500">
              Low: Fast data synthesis. Medium: Balanced synthesis and speed. High: Rich
              speed.
            </span>
          </div>
        </div>

        {/* Step 3: Choose Base Model for Training Second Me */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <h4 className="text-base font-semibold text-gray-800 mb-1">
              Step 3: Choose Base Model for Training Second Me
            </h4>
            <span className="text-xs text-gray-500">Base model for training your Second Me.</span>
          </div>
          {loadingModels ? (
            <div className="flex justify-center items-center h-10">
              <Spin />
            </div>
          ) : (
            <Listbox
              disabled={disabledChangeParams || !hasCloudServiceApiKey || loadingModels}
              onChange={(value) => {
                // 检查是否真的发生了变化，避免不必要的更新
                if (value !== trainingParams.model_name) {
                  updateTrainingParams({ 
                    ...trainingParams, 
                    model_name: value,
                    cloud_model_name: value 
                  });
                }
              }}
              value={trainingParams.model_name || trainingParams.cloud_model_name || ''}
            >
              <div className="relative mt-1">
                <Listbox.Button
                  className={classNames(
                    'relative w-full cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300',
                    (disabledChangeParams || !hasCloudServiceApiKey || loadingModels) ? 'opacity-50 !cursor-not-allowed' : 'hover:bg-gray-50'
                  )}
                >
                  <span className="block truncate">
                    {(() => {
                      // 尝试使用 model_name 查找选项
                      const currentOption = cloudModelOptions.find(
                        (option) => option.value === trainingParams.model_name
                      );
                      
                      // 如果没找到，尝试使用 cloud_model_name 查找
                      if (!currentOption && trainingParams.cloud_model_name) {
                        const cloudOption = cloudModelOptions.find(
                          (option) => option.value === trainingParams.cloud_model_name
                        );
                        if (cloudOption) return cloudOption.label;
                      }
                      
                      // 返回找到的选项标签，或默认文本
                      return currentOption?.label || 
                             (cloudModelOptions.length > 0 
                               ? cloudModelOptions[0].label 
                               : 'No models available');
                    })()}
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
                    {cloudModelOptions.length === 0 ? (
                      <div className="py-2 px-4 text-center text-sm text-gray-500">
                        No models available. Check API key or provider.
                      </div>
                    ) : (
                      cloudModelOptions.map((option) => (
                        <Listbox.Option
                          key={option.value}
                          className={({ active }) =>
                            `relative cursor-pointer select-none py-2 pl-10 pr-4 ${active ? 'bg-blue-100 text-blue-900' : 'text-gray-900'}`
                          }
                          value={option.value}
                        >
                          {({ selected }) => {
                            // 额外检查 selected 状态是否与当前值匹配
                            // 同时考虑 model_name 和 cloud_model_name
                            const isSelected = selected || 
                                              trainingParams.model_name === option.value || 
                                              trainingParams.cloud_model_name === option.value;
                            return (
                              <>
                                <span
                                  className={`block truncate ${isSelected ? 'font-medium' : 'font-normal'}`}
                                >
                                  {option.label}
                                </span>
                                {isSelected ? (
                                  <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-blue-600">
                                    <DoneIcon className="h-5 w-5" />
                                  </span>
                                ) : null}
                              </>
                            );
                          }}
                        </Listbox.Option>
                      ))
                    )}
                  </Listbox.Options>
                </Transition>
              </div>
            </Listbox>
          )}
          {!hasCloudServiceApiKey && (
            <p className="text-xs text-red-500">Please configure Cloud Provider in Step 1 to see available models.</p>
          )}
          {hasCloudServiceApiKey && !loadingModels && cloudModelOptions.length === 0 && (
            <p className="text-xs text-orange-500">No base models available for the selected cloud provider or API key.</p>
          )}
        </div>

        {/* Step 4: Configure Advanced Training Parameters */}
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
                  if (value == null) return;
                  updateTrainingParams({ ...trainingParams, learning_rate: value });
                }}
                status={
                  trainingParams.learning_rate === 0.005 || trainingParams.learning_rate === 0.00003
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
                  if (value == null) return;
                  updateTrainingParams({ ...trainingParams, number_of_epochs: value });
                }}
                status={
                  trainingParams.number_of_epochs === 10 || trainingParams.number_of_epochs === 1
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
          </div>
        </div>
      </div>

      <ThinkingModelModal onClose={() => setOpenThinkingModel(false)} open={openThinkingModel} />
      <CloudProviderModal
        open={openCloudProviderModal}
        onClose={() => setOpenCloudProviderModal(false)}
        cloudConfig={cloudConfig} // Pass current cloudConfig from store
        updateCloudConfig={updateCloudConfig} // Pass update function from store
        saveCloudConfig={async () => { // Implement saveCloudConfig to align with modal props
          // This function in the modal is expected to save the config (e.g., call an API)
          // Here, we assume the modal itself handles the API call via its internal handleUpdate
          // and then calls this callback. We just need to ensure the local state/store is updated.
          fetchCloudConfig(); // Re-fetch config to update the store and related states
        }}
      />
    </div>
  );
};

export default CloudTrainingConfig;
