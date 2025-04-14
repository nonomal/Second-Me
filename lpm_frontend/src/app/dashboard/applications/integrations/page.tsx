import { CalendarOutlined } from '@ant-design/icons';

const Page = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] px-5 py-7">
      <div className="text-center">
        <div className="flex justify-center mb-6">
          <CalendarOutlined style={{ fontSize: '64px', color: '#1677ff' }} />
        </div>
        <h1 className="text-3xl font-bold mb-4">Coming Soon</h1>
        <p className="text-lg text-gray-600 mb-6">
          We&apos;re working hard to bring you amazing integration features.
        </p>
        <p className="text-md text-gray-500">Stay tuned for updates!</p>
      </div>
    </div>
  );
};

export default Page;
