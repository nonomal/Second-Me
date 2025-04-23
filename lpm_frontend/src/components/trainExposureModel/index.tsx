import { getStepOutputContent } from '@/service/train';
import { message, Modal } from 'antd';
import { useEffect, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface IProps {
  handleClose: () => void;
  stepName: string;
}

const obj = {
  columns: [
    'id',
    'human_readable_id',
    'title',
    'type',
    'description',
    'text_unit_ids',
    'frequency',
    'degree',
    'x',
    'y'
  ],
  content: [
    {
      degree: 3,
      description:
        '\u571f\u8033\u5176\u662f\u4e00\u4e2a\u56fd\u5bb6\uff0c\u6700\u8fd1\u5bf9\u4e2d\u56fd\u7535\u52a8\u6c7d\u8f66\u589e\u52a0\u4e86\u989d\u5916\u5173\u7a0e',
      frequency: 1,
      human_readable_id: 0,
      id: 'dee8f22f-be0f-43ae-91c3-6cd5f643031d',
      text_unit_ids: [
        '92ba7fb87db25d3c432a703a5a3518126b98d066bb590f8191c8a526e00f36fd6cfed1277aaa303a772dbd3419d7ec0ca63952f716886063b7dfba25d36ac78c'
      ],
      title: '\u571f\u8033\u5176',
      type: 'GEO',
      x: 0.0,
      y: 0.0
    },
    {
      degree: 3,
      description:
        '\u4e2d\u56fd\u7535\u52a8\u6c7d\u8f66\u662f\u4e2d\u56fd\u5236\u9020\u7684\u7535\u52a8\u6c7d\u8f66\uff0c\u6700\u8fd1\u53d7\u5230\u571f\u8033\u5176\u989d\u5916\u5173\u7a0e\u7684\u5f71\u54cd',
      frequency: 1,
      human_readable_id: 1,
      id: '8bde84ce-98d8-419f-9350-a8de559bed2a',
      text_unit_ids: [
        '92ba7fb87db25d3c432a703a5a3518126b98d066bb590f8191c8a526e00f36fd6cfed1277aaa303a772dbd3419d7ec0ca63952f716886063b7dfba25d36ac78c'
      ],
      title: '\u4e2d\u56fd\u7535\u52a8\u6c7d\u8f66',
      type: 'SPECIFIC OBJECT',
      x: 0.0,
      y: 0.0
    }
  ]
};

const TrainExposureModel = (props: IProps) => {
  const { handleClose, stepName } = props;
  const [outputContent, setOutputContent] = useState('');

  useEffect(() => {
    if (!stepName) return;

    setOutputContent('');

    getStepOutputContent(stepName).then((res) => {
      if (res.data.code == 0) {
        const data = res.data.data;

        const formattedData = JSON.stringify(data, null, 2);

        setOutputContent(formattedData);
      } else {
        console.error(res.data.message);
      }
    });
  }, [stepName]);

  const renderOutputContent = () => {
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
        {outputContent}
      </SyntaxHighlighter>
    );
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
      <div className="bg-[#f5f5f5] border max-h-[600px] overflow-scroll border-[#e0e0e0] rounded p-4 font-mono text-sm leading-6 text-[#333] shadow-sm transition-all duration-300 ease-in-out w-full max-w-full">
        {renderOutputContent()}
      </div>
    </Modal>
  );
};

export default TrainExposureModel;
