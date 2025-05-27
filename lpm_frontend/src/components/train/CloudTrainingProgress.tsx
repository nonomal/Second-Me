import type { TrainingJobInfo } from '@/service/cloudService';
import type { IStepOutputInfo } from '../trainExposureModel';
import TrainExposureModel from '../trainExposureModel';
import { useState } from 'react';
import classNames from 'classnames';

interface CloudTrainingProgressProps {
  status: string;
  cloudJobInfo: TrainingJobInfo | null;
}

// Cloud training stages descriptions (excluding "Downloading the Base Model")
const descriptionMap = [
  "This step starts by processing and organizing your memories into a structured digital format that forms the groundwork for your Second Me. We break down your life experiences into smaller, meaningful pieces, encode them systematically, and extract essential insights to create a solid base. It's the first move toward building an entity that reflects your past and present.",
  "Here, we take the fragments of your memories and weave them into a complete, flowing biography that captures your essence. This process connects the dots between your experiences, shaping them into a coherent story that defines who you are. It's like crafting the blueprint of a new being born from your life's journey.",
  "To enable your Second Me to understand you fully, we create specialized training data tailored to your unique profile. This step lays the groundwork for it to grasp your preferences, identity, and knowledge accurately, ensuring the entity we're constructing can think and respond in ways that feel authentic to you.",
  'Finally, we train the core model with your specific memories, traits, and preferences, blending them seamlessly into its framework. This step transforms the model into a living representation of you, merging technology with your individuality to create a Second Me that feels real and true to your essence.'
];

const CloudTrainingProgress = (props: CloudTrainingProgressProps): JSX.Element => {
  const { status, cloudJobInfo } = props;

  const [stepOutputInfo, setStepOutputInfo] = useState<IStepOutputInfo>({} as IStepOutputInfo);

  const formatUnderscoreToName = (_str: string) => {
    const str = _str || '';

    return str
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const formatToUnderscore = (str: string): string => {
    if (!str) return '';

    return str.toLowerCase().replace(/\s+/g, '_');
  };

  // Create cloud training stages (same structure as local but without "Downloading the Base Model")
  const getCloudTrainingStages = () => {
    const getStageStatus = (stageIndex: number) => {
      if (status === 'trained') return 'completed';
      
      if (!cloudJobInfo) {
        return stageIndex === 0 ? 'in_progress' : 'pending';
      }

      switch (cloudJobInfo.status) {
        case 'PENDING':
          return stageIndex === 0 ? 'in_progress' : 'pending';
        case 'RUNNING':
          if (stageIndex <= 1) return 'completed';

          return stageIndex === 2 ? 'in_progress' : 'pending';
        case 'SUCCEEDED':
          return 'completed';
        case 'FAILED':
          return stageIndex === 0 ? 'failed' : 'pending';
        default:
          return stageIndex === 0 ? 'in_progress' : 'pending';
      }
    };

    const getStageProgress = (stageIndex: number) => {
      const stageStatus = getStageStatus(stageIndex);

      if (stageStatus === 'completed') return 100;

      if (stageStatus === 'failed') return 0;

      if (stageStatus === 'in_progress') {
        // Progressive progress based on stage and job status
        if (stageIndex === 0) return cloudJobInfo ? 80 : 30;

        if (stageIndex === 1) return 60;

        if (stageIndex === 2) return 70;

        if (stageIndex === 3) return 85;
      }

      return 0;
    };

    const getCurrentStep = (stageIndex: number) => {
      const stageStatus = getStageStatus(stageIndex);

      if (stageStatus !== 'in_progress') return null;

      // Return appropriate step based on stage
      const stepMaps = [
        ['list_documents', 'generate_document_embeddings', 'process_chunks', 'chunk_embedding'],
        ['extract_dimensional_topics', 'map_your_entity_network'],
        ['decode_preference_patterns', 'reinforce_identity', 'augment_content_retention'],
        ['train', 'merge_weights', 'convert_model']
      ];

      return stepMaps[stageIndex] ? stepMaps[stageIndex][0] : null;
    };

    const stages = [
      {
        name: 'Activating the Memory Matrix',
        description: descriptionMap[0],
        status: getStageStatus(0),
        progress: getStageProgress(0),
        current_step: getCurrentStep(0),
        steps: [
          {
            completed: getStageStatus(0) === 'completed',
            name: 'List Documents',
            status:
              getStageStatus(0) === 'completed'
                ? 'completed'
                : getStageStatus(0) === 'in_progress'
                  ? 'in_progress'
                  : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(0) === 'completed',
            name: 'Generate Document Embeddings',
            status: getStageStatus(0) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(0) === 'completed',
            name: 'Process Chunks',
            status: getStageStatus(0) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(0) === 'completed',
            name: 'Chunk Embedding',
            status: getStageStatus(0) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          }
        ]
      },
      {
        name: 'Synthesize Your Life Narrative',
        description: descriptionMap[1],
        status: getStageStatus(1),
        progress: getStageProgress(1),
        current_step: getCurrentStep(1),
        steps: [
          {
            completed: getStageStatus(1) === 'completed',
            name: 'Extract Dimensional Topics',
            status:
              getStageStatus(1) === 'completed'
                ? 'completed'
                : getStageStatus(1) === 'in_progress'
                  ? 'in_progress'
                  : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(1) === 'completed',
            name: 'Map Your Entity Network',
            status: getStageStatus(1) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          }
        ]
      },
      {
        name: 'Prepare Training Data for Deep Comprehension',
        description: descriptionMap[2],
        status: getStageStatus(2),
        progress: getStageProgress(2),
        current_step: getCurrentStep(2),
        steps: [
          {
            completed: getStageStatus(2) === 'completed',
            name: 'Decode Preference Patterns',
            status:
              getStageStatus(2) === 'completed'
                ? 'completed'
                : getStageStatus(2) === 'in_progress'
                  ? 'in_progress'
                  : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(2) === 'completed',
            name: 'Reinforce Identity',
            status: getStageStatus(2) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(2) === 'completed',
            name: 'Augment Content Retention',
            status: getStageStatus(2) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          }
        ]
      },
      {
        name: 'Training to create Second Me',
        description: descriptionMap[3],
        status: getStageStatus(3),
        progress: getStageProgress(3),
        current_step: getCurrentStep(3),
        steps: [
          {
            completed: getStageStatus(3) === 'completed',
            name: 'Train',
            status:
              getStageStatus(3) === 'completed'
                ? 'completed'
                : getStageStatus(3) === 'in_progress'
                  ? 'in_progress'
                  : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(3) === 'completed',
            name: 'Merge Weights',
            status: getStageStatus(3) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          },
          {
            completed: getStageStatus(3) === 'completed',
            name: 'Convert Model',
            status: getStageStatus(3) === 'completed' ? 'completed' : 'pending',
            have_output: false,
            path: ''
          }
        ]
      }
    ];

    return stages;
  };

  const trainingStages = getCloudTrainingStages();

  // Calculate overall progress
  const calculateOverallProgress = () => {
    if (status === 'trained') return 100;

    const totalStages = trainingStages.length;
    const totalProgress = trainingStages.reduce((sum, stage) => sum + stage.progress, 0);

    return totalProgress / totalStages;
  };

  const overallProgress = calculateOverallProgress();

  // Get current stage
  const getCurrentStageName = () => {
    const currentStage = trainingStages.find((stage) => stage.status === 'in_progress');

    return currentStage ? formatToUnderscore(currentStage.name) : '';
  };

  const currentStage = getCurrentStageName();

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">
          Cloud Training Progress (faster training with pre-loaded models)
        </h3>
        {status === 'trained' && (
          <span className="px-2.5 py-1 bg-green-50 text-green-700 text-sm font-medium rounded-full">
            Training Complete
          </span>
        )}
      </div>
      <div className="space-y-6">
        {/* Overall Progress */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-lg font-semibold text-gray-900">Overall Progress</span>
              <span className="text-2xl font-bold text-purple-600">
                {Math.round(overallProgress)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>
        </div>

        {/* Cloud Job Info */}
        {cloudJobInfo && (
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-purple-900">Job ID</span>
                <span className="text-sm text-purple-700">{cloudJobInfo.job_id}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-purple-900">Status</span>
                <span
                  className={`text-sm px-2 py-1 rounded-full ${
                    cloudJobInfo.status === 'SUCCEEDED'
                      ? 'bg-green-100 text-green-700'
                      : cloudJobInfo.status === 'RUNNING'
                        ? 'bg-blue-100 text-blue-700'
                        : cloudJobInfo.status === 'FAILED'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-yellow-100 text-yellow-700'
                  }`}
                >
                  {cloudJobInfo.status}
                </span>
              </div>
              {cloudJobInfo.timestamp && (
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-purple-900">Created</span>
                  <span className="text-sm text-purple-700">
                    {new Date(cloudJobInfo.timestamp).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* All Training Stages */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-700">Training Stages</h4>
          <div className="space-y-4">
            {trainingStages.map((stage) => {
              const stageStatus = stage.status;
              const progress = stage.progress;

              // Handle NaN case
              const displayProgress = isNaN(progress) ? 0 : progress;

              const isCurrentStage = formatUnderscoreToName(currentStage) == stage.name;

              return (
                <div
                  key={stage.name}
                  className="bg-white rounded-lg border border-gray-100 p-4 shadow-sm"
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="flex-shrink-0">
                      {stageStatus === 'completed' ? (
                        <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center">
                          <svg
                            className="w-4 h-4 text-green-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              d="M5 13l4 4L19 7"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                            />
                          </svg>
                        </div>
                      ) : stageStatus === 'in_progress' ? (
                        <div className="w-6 h-6 rounded-full bg-purple-100 flex items-center justify-center">
                          <div className="w-3 h-3 rounded-full bg-purple-600 animate-pulse" />
                        </div>
                      ) : (
                        <div className="w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center">
                          <div className="w-3 h-3 rounded-full bg-gray-300" />
                        </div>
                      )}
                    </div>

                    <div className="flex-grow">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center">
                          <span
                            className={`text-sm font-medium ${
                              isCurrentStage ? 'text-purple-700' : 'text-gray-700'
                            }`}
                          >
                            {stage.name}
                            {isCurrentStage && stage.current_step && (
                              <span className="ml-2 text-xs text-gray-500">
                                {formatUnderscoreToName(stage.current_step)}
                              </span>
                            )}
                          </span>
                          <button
                            className="ml-1 p-1 rounded-full text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
                            onClick={() => {
                              const modal = document.createElement('div');

                              modal.className =
                                'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
                              modal.innerHTML = `
                                <div class="bg-white rounded-xl max-w-md p-6 m-4 space-y-4 relative shadow-xl">
                                  <h3 class="text-xl font-semibold">${stage.name}</h3>
                                  <div class="space-y-4 text-gray-600">
                                    <p>${stage.description}</p>
                                  </div>
                                  <button class="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600" onclick="this.parentElement.parentElement.remove()">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                  </button>
                                </div>
                              `;
                              document.body.appendChild(modal);
                              modal.onclick = (e) => {
                                if (e.target === modal) modal.remove();
                              };
                            }}
                            title="Learn more about this stage"
                          >
                            <svg
                              className="w-3 h-3"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                              />
                            </svg>
                          </button>
                        </div>
                        <span className="text-xs text-gray-500">
                          {Math.round(displayProgress)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-100 rounded-full h-1.5 mt-1">
                        <div
                          className={`h-1.5 rounded-full transition-all duration-500 ${
                            stageStatus === 'completed'
                              ? 'bg-green-500'
                              : stageStatus === 'in_progress'
                                ? 'bg-purple-500'
                                : displayProgress > 0
                                  ? 'bg-purple-300'
                                  : 'bg-gray-200'
                          }`}
                          style={{ width: `${displayProgress}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Step list */}
                  <div className="mt-3 pl-9">
                    {stage.steps.length > 0 ? (
                      <div className="space-y-2">
                        {stage.steps.map((step, stepIndex) => (
                          <div key={stepIndex} className="flex items-center space-x-2">
                            <div className="flex-shrink-0">
                              {step.completed ? (
                                <div className="w-4 h-4 rounded-full bg-green-100 flex items-center justify-center">
                                  <svg
                                    className="w-3 h-3 text-green-600"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      d="M5 13l4 4L19 7"
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth="2"
                                    />
                                  </svg>
                                </div>
                              ) : stage.current_step &&
                                formatUnderscoreToName(stage.current_step) == step.name ? (
                                <div className="w-4 h-4 rounded-full bg-purple-100 flex items-center justify-center">
                                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                                </div>
                              ) : (
                                <div className="w-4 h-4 rounded-full bg-gray-100 flex items-center justify-center">
                                  <div className="w-2 h-2 rounded-full bg-gray-300" />
                                </div>
                              )}
                            </div>
                            <span
                              className={classNames(
                                'text-xs',
                                stage.current_step &&
                                  formatUnderscoreToName(stage.current_step) == step.name
                                  ? 'text-purple-600 font-medium'
                                  : 'text-gray-600'
                              )}
                            >
                              {step.name}
                            </span>

                            {step.completed && step.have_output && (
                              <span
                                className="text-xs text-purple-500 underline cursor-pointer hover:text-purple-600"
                                onClick={() => {
                                  setStepOutputInfo({
                                    stepName: formatToUnderscore(step.name),
                                    path: step.path
                                  });
                                }}
                              >
                                View Resources
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : stageStatus !== 'pending' ? (
                      <div className="text-xs text-gray-500">Processing...</div>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <TrainExposureModel
        handleClose={() => setStepOutputInfo({} as IStepOutputInfo)}
        stepOutputInfo={stepOutputInfo}
      />
    </div>
  );
};

export default CloudTrainingProgress;
