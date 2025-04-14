'use client';

import { useEffect, useState } from 'react';
import Content from './content';
import Tools from './tools';
import './markdown.css';
import { useSSE } from '@/hooks/useSSE';
import { Segmented } from 'antd';

type responseTypeOptions = 'raw' | 'text';

const MCPMode = () => {
  const [activeTab, setActiveTab] = useState('content');
  const [chosenTool, setChosenTool] = useState('');
  const [instanceId, setInstanceId] = useState('');
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [rawResponse, setRawResponse] = useState('');
  const [responseType, setResponseType] = useState<responseTypeOptions>('raw');
  const { sendStreamMessage, streamContent, streamRawContent, streaming } = useSSE();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!instanceId.trim() || !query.trim()) {
      return;
    }

    sendStreamMessage({
      messages: [
        {
          role: 'system',
          content: ''
        },
        {
          role: 'user',
          content: query
        }
      ],
      metadata: {
        enable_l0_retrieval: true,
        enable_l1_retrieval: true
      },
      temperature: 0.7,
      stream: true
    });
  };

  useEffect(() => {
    if (streaming) {
      setResponse(streamContent);
    }
  }, [streaming, streamContent]);

  useEffect(() => {
    if (streaming) {
      setRawResponse(streamRawContent);
    }
  }, [streaming, streamRawContent]);

  const items = [
    {
      key: 'content',
      label: 'Content',
      children: <Content />
    },
    {
      key: 'tools',
      label: 'Tools',
      children: <Tools chosenTool={chosenTool} setChosenTool={setChosenTool} />
    }
  ];

  return (
    <div className="flex gap-4 p-4 justify-between w-full">
      <div className="w-[calc(66%-16px)] h-fit">
        <div className="shadow-md rounded-lg h-full border border-gray-200 bg-white">
          <div className="border-b border-gray-200">
            <div className="flex">
              {items.map((item) => (
                <div
                  key={item.key}
                  className={`px-4 py-2 cursor-pointer ${activeTab === item.key ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
                  onClick={() => setActiveTab(item.key)}
                >
                  {item.label}
                </div>
              ))}
            </div>
          </div>
          <div className="overflow-scroll max-h-[60vh] markdown-content">
            <div className="max-w-full h-fit">
              {items.find((item) => item.key === activeTab)?.children}
            </div>
          </div>
        </div>
      </div>
      {chosenTool && (
        <div className="w-1/3 shrink-0 h-fit">
          <div className="shadow-md rounded-lg h-full p-4 border border-gray-200 bg-white">
            <div className="text-lg font-medium mb-4">Test API</div>

            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label
                  className="block text-sm font-medium text-gray-700 mb-1"
                  htmlFor="instanceId"
                >
                  Instance ID
                </label>
                <input
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  id="instanceId"
                  onChange={(e) => setInstanceId(e.target.value)}
                  placeholder="Enter your instance ID"
                  type="text"
                  value={instanceId}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="query">
                  Query
                </label>
                <input
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  id="query"
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your query"
                  type="text"
                  value={query}
                />
              </div>

              <button
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!instanceId.trim() || !query.trim() || streaming}
                type="submit"
              >
                Submit
              </button>
            </form>

            {(response || rawResponse) && (
              <>
                <Segmented
                  className="mt-4"
                  onChange={(value) => setResponseType(value as responseTypeOptions)}
                  options={['raw', 'text']}
                  value={responseType}
                />
                <div className="mt-6">
                  <div className="text-sm font-medium text-gray-700 mb-2">Response:</div>
                  <div className="bg-gray-50 border border-gray-200 rounded-md p-3 whitespace-pre-wrap text-sm font-mono max-h-[300px] overflow-auto">
                    {responseType === 'raw' ? rawResponse : response}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MCPMode;
