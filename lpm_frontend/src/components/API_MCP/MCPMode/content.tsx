import SimplyMD from '@/components/SimpleMD';

const Content = () => {
  return (
    <div className="p-4">
      <SimplyMD
        content={`### 123
      [ ] some`}
      />
    </div>
  );
};

export default Content;
