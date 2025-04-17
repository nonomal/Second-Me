import MarkdownContent from '@/utils/markdown';
import { useEffect, useState } from 'react';
import type { ReactElement } from 'react';

const Content = (): ReactElement => {
  const [markdownContent, setMarkdownContent] = useState<string>('');

  useEffect(() => {
    fetch('/docs/mcp_content.md')
      .then((response) => response.text())
      .then((text) => setMarkdownContent(text));
  }, []);

  return (
    <div className="p-4 h-full w-full markdown-wrapper">
      <MarkdownContent markdownContent={markdownContent} />
    </div>
  );
};

export default Content;
