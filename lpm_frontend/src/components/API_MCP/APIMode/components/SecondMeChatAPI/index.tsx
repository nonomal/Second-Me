'use client';

import { Badge, Button, Input } from 'antd';
import { CheckCircleFilled, CopyOutlined } from '@ant-design/icons';
import { useState, useMemo } from 'react';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';

const endpoint = '/api/chat/{instance_id}/chat/completions';

const SecondMeChatAPI = () => {
  const [copied, setCopied] = useState(false);
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);

  const handleCopyEndpoint = () => {
    navigator.clipboard.writeText(endpoint);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4">
      {/* API Status Card */}
      <div className="bg-white border border-solid border-gray-200 rounded-md shadow-sm p-4">
        <div className="text-base font-medium mb-2">Second Me Chat API</div>
        <div className="flex items-center">
          <Badge status={isRegistered ? 'success' : 'error'} />
          {isRegistered ? (
            <div className="ml-2 text-[#2bee9d] font-medium">In Service</div>
          ) : (
            <div className="ml-2 text-[#ff4d4f] font-medium">Not In Service</div>
          )}
        </div>
      </div>

      {/* API Endpoint Card */}
      <div className="bg-white border border-solid border-gray-200 rounded-md shadow-sm p-4">
        <div className="text-base font-medium mb-2">Service API Endpoint</div>
        <div className="flex items-center mb-2">
          <Input
            className="font-mono"
            readOnly
            suffix={
              <div
                className="ml-2 cursor-pointer flex items-center justify-center hover:bg-gray-100 p-1 rounded-md transition-colors"
                onClick={handleCopyEndpoint}
              >
                {copied ? <CheckCircleFilled className="text-green-500" /> : <CopyOutlined />}
              </div>
            }
            value={endpoint}
          />
        </div>
        <div className="text-base font-medium mb-3 mt-4">API Management</div>
        <div className="flex flex-wrap gap-4">
          <a
            href="https://github.com/mindverse/Second-Me/blob/master/docs/Public%20Chat%20API.md"
            rel="noreferrer"
            target="_blank"
          >
            <Button
              className="min-w-[160px] flex items-center justify-center"
              icon={<span className="mr-2">📄</span>}
              size="large"
            >
              API Reference
            </Button>
          </a>
        </div>
      </div>
    </div>
  );
};

export default SecondMeChatAPI;
