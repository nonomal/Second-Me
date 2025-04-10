'use client';

import { Button, Space } from 'antd';
import { useState } from 'react';
import SecondMeChatAPI from './components/SecondMeChatAPI';
import BridgeModeAPI from './components/BridgeModeAPI';

const APIMode = () => {
  const [activeTab, setActiveTab] = useState('1');

  const handleTabChange = (key: string) => {
    setActiveTab(key);
  };

  return (
    <div className="w-full">
      <div className="mb-6">
        <Space className="flex justify-center sm:justify-start" size="middle">
          <Button
            className="min-w-[180px] h-10 flex items-center justify-center"
            onClick={() => handleTabChange('1')}
            type={activeTab === '1' ? 'primary' : 'default'}
          >
            Second Me Chat API
          </Button>
          <Button
            className="min-w-[180px] h-10 flex items-center justify-center"
            onClick={() => handleTabChange('2')}
            type={activeTab === '2' ? 'primary' : 'default'}
          >
            <span>Bridge Mode API</span>
            <span className="ml-2 text-xs text-blue-500">(Coming Soon)</span>
          </Button>
        </Space>
      </div>

      <div className="mt-4">
        {activeTab === '1' && <SecondMeChatAPI />}
        {activeTab === '2' && <BridgeModeAPI />}
      </div>
    </div>
  );
};

export default APIMode;
