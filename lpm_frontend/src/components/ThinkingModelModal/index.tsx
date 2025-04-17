import { Input, Modal } from 'antd';
import { useState } from 'react';

interface IProps {
  open: boolean;
  onClose: () => void;
}

const ThinkingModelModal = (props: IProps) => {
  const { open, onClose: handleCancel } = props;
  const [thinkParams, setThinkParams] = useState({
    model_name: '',
    api_key: '',
    api_endpoint: ''
  });

  return (
    <Modal centered onCancel={handleCancel} open={open}>
      <div className="flex flex-col gap-2 mb-4">
        <div className="text-xl leading-6 font-semibold text-gray-900">Thinking model</div>
        <div className="text-sm font-medium text-gray-700">Currently only supports DeepSeek</div>
      </div>
      <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model Name</label>
            <Input
              autoComplete="off"
              className="w-full"
              onChange={(e) => setThinkParams({ ...thinkParams, model_name: e.target.value })}
              value={thinkParams.model_name}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
            {/* form is to disable autoComplete */}
            <form autoComplete="off">
              <Input.Password
                autoComplete="off"
                className="w-full"
                onChange={(e) => setThinkParams({ ...thinkParams, api_key: e.target.value })}
                value={thinkParams.api_key}
              />
            </form>
          </div>
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">API Endpoint</label>
          <Input
            autoComplete="off"
            className="w-full"
            onChange={(e) => setThinkParams({ ...thinkParams, api_endpoint: e.target.value })}
            value={thinkParams.api_endpoint}
          />
        </div>
      </div>
    </Modal>
  );
};

export default ThinkingModelModal;
