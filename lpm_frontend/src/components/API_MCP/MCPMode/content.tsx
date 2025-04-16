import { useEffect, useState } from 'react';
import Markdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { ReactElement } from 'react';

interface CodeProps {
  node?: any;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

const CodeBlock = ({ node, inline, className, children, ...props }: CodeProps): ReactElement => {
  const match = /language-(\w+)/.exec(className || '');

  return !inline && match ? (
    <SyntaxHighlighter PreTag="div" language={match[1]} style={oneDark} {...props}>
      {String(children).replace(/\n$/, '')}
    </SyntaxHighlighter>
  ) : (
    <code className={className} {...props}>
      {children}
    </code>
  );
};

const Content = (): ReactElement => {
  const [markdownContent, setMarkdownContent] = useState<string>('');

  useEffect(() => {
    fetch('/docs/mcp_content.md')
      .then((response) => response.text())
      .then((text) => setMarkdownContent(text));
  }, []);

  return (
    <div className="p-4 h-full w-full markdown-wrapper">
      <Markdown
        components={{
          code: CodeBlock
        }}
      >
        {markdownContent}
      </Markdown>
    </div>
  );
};

export default Content;
