import MCPMode from '@/components/API_MCP/MCPMode';
import APIMode from '@/components/API_MCP/APIMode';
import { Tabs } from 'antd';

const items = [
  {
    key: 'api',
    label: 'API',
    children: <APIMode />
  },
  {
    key: 'mcp',
    label: 'MCP',
    children: <MCPMode />
  }
];

const Page = () => {
  return (
    <div className="px-5 py-7 w-full h-[calc(100vh-64px)] overflow-scroll">
      <div className="flex flex-col mb-10">
        <div className="font-extrabold text-2xl">Second Me Services</div>
        <div>Manage your API and intergration services</div>
      </div>
      <Tabs className="w-full" defaultActiveKey="api" items={items} />
    </div>
  );
};

export default Page;
