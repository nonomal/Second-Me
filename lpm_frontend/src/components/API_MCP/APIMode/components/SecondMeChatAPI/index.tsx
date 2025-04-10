'use client';

import { Badge, Button, Input, Typography } from 'antd';
import { CheckCircleFilled, CopyOutlined } from '@ant-design/icons';
import { useState } from 'react';

const { Paragraph } = Typography;

const SecondMeChatAPI = () => {
  const [copied, setCopied] = useState(false);

  // Mock data - would come from API in real implementation
  const apiStatus = {
    online: true,
    endpoint: 'https://api.secondme.ai/v1/chat'
  };

  const handleCopyEndpoint = () => {
    navigator.clipboard.writeText(apiStatus.endpoint);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4">
      {/* API Status Card */}
      <div className="bg-white border border-solid border-gray-200 rounded-md shadow-sm p-4">
        <div className="text-base font-medium mb-2">Second Me Chat API</div>
        <div className="flex items-center">
          <Badge status={apiStatus.online ? 'success' : 'error'} />
          {apiStatus.online ? (
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
            value={apiStatus.endpoint}
          />
        </div>
      </div>

      {/* Action Buttons Card */}
      <div className="bg-white border border-solid border-gray-200 rounded-md shadow-sm p-4">
        <div className="text-base font-medium mb-3">API Management</div>
        <div className="flex flex-wrap gap-4">
          <Button
            className="min-w-[160px] flex items-center justify-center"
            icon={<span className="mr-2">🔑</span>}
            size="large"
          >
            Manage API Keys
          </Button>
          <Button
            className="min-w-[160px] flex items-center justify-center"
            icon={<span className="mr-2">📄</span>}
            size="large"
          >
            API Reference
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SecondMeChatAPI;
