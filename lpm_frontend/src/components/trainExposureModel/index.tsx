import type { TrainStepOutput } from '@/service/train';
import { getStepOutputContent } from '@/service/train';
import { Modal, Table } from 'antd';
import { useEffect, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface IProps {
  handleClose: () => void;
  stepName: string;
}

const TrainExposureModel = (props: IProps) => {
  const { handleClose, stepName } = props;
  const [outputContent, setOutputContent] = useState<TrainStepOutput | null>(null);

  useEffect(() => {
    if (!stepName) return;

    setOutputContent(null);

    getStepOutputContent(stepName).then((res) => {
      if (res.data.code == 0) {
        const data = res.data.data;

        setOutputContent(data);
      } else {
        console.error(res.data.message);
      }
    });
  }, [stepName]);

  const renderOutputContent = () => {
    if (!outputContent || typeof outputContent !== 'object') return null;

    if (outputContent.file_type == 'json') {
      const showContent = JSON.stringify(outputContent.content, null, 2);

      return (
        <SyntaxHighlighter
          customStyle={{
            backgroundColor: 'transparent',
            margin: 0,
            padding: 0
          }}
          language="json"
          style={tomorrow}
        >
          {showContent}
        </SyntaxHighlighter>
      );
    }

    if (outputContent.file_type == 'parquet') {
      const columns = outputContent.columns.map((item, index) => ({
        title: item,
        dataIndex: item,
        key: index
      }));
      const data = outputContent.content;

      return (
        <Table className="w-fit max-w-fit" columns={columns} dataSource={data} pagination={false} />
      );
    }

    return null;
  };

  return (
    <Modal
      centered
      closable={false}
      footer={null}
      onCancel={handleClose}
      open={!!stepName}
      width={800}
    >
      <div className="bg-[#f5f5f5] max-h-[600px] w-full overflow-scroll border border-[#e0e0e0] rounded p-4 font-mono text-sm leading-6 text-[#333] shadow-sm transition-all duration-300 ease-in-out">
        {renderOutputContent()}
      </div>
    </Modal>
  );
};

export default TrainExposureModel;
