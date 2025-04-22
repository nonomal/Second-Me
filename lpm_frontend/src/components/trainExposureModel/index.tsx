import { getStepOutputContent } from '@/service/train';
import { message, Modal } from 'antd';
import { useEffect, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface IProps {
  handleClose: () => void;
  stepName: string;
}

const TrainExposureModel = (props: IProps) => {
  const { handleClose, stepName } = props;
  const [outputContent, setOutputContent] = useState('');

  useEffect(() => {
    if (!stepName) return;

    getStepOutputContent(stepName).then((res) => {
      if (res.data.code === 0) {
        const data = res.data.data.content || '';

        try {
          const formattedData =
            typeof data === 'string'
              ? JSON.stringify(JSON.parse(data), null, 2)
              : JSON.stringify(data, null, 2);

          setOutputContent(formattedData);
        } catch {
          setOutputContent(JSON.stringify(data));
        }
      } else {
        console.error(res.data.message);
      }
    });
  }, [stepName]);

  return (
    <Modal
      centered
      closable={false}
      footer={null}
      onCancel={handleClose}
      open={!!stepName}
      width={800}
    >
      <div className="bg-[#f5f5f5] border max-h-[600px] overflow-scroll border-[#e0e0e0] rounded p-4 font-mono text-sm leading-6 overflow-x-auto text-[#333] shadow-sm transition-all duration-300 ease-in-out w-full max-w-full">
        <SyntaxHighlighter
          customStyle={{
            backgroundColor: 'transparent',
            margin: 0,
            padding: 0
          }}
          language="json"
          style={tomorrow}
        >
          {outputContent}
        </SyntaxHighlighter>
      </div>
    </Modal>
  );
};

export default TrainExposureModel;
