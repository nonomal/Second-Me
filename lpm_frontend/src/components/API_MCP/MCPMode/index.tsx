'use client';

import { useState } from 'react';
import Content from './content';

const items = [
  {
    key: 'content',
    label: 'Content',
    children: <Content />
  },
  {
    key: 'tools',
    label: 'Tools',
    children: <div className="p-4">tools</div>
  }
];

const MCPMode = () => {
  const [activeTab, setActiveTab] = useState('content');

  return (
    <div className="flex gap-4 p-4 w-full">
      <div className="flex-1 min-w-0">
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
          <div className="w-full">{items.find((item) => item.key === activeTab)?.children}</div>
        </div>
      </div>
      <div className="w-1/3">
        <div className="shadow-md rounded-lg h-full p-4 border border-gray-200 bg-white">
          <div className="text-lg font-medium mb-3">Information</div>
          <div className="text-gray-600">
            Additional information and details can be displayed here.
          </div>
        </div>
      </div>
    </div>
  );
};

export default MCPMode;
