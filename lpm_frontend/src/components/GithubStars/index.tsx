'use client';

import { useEffect, useState } from 'react';
import { GithubOutlined } from '@ant-design/icons';
import { formatNumber } from '@/utils/formatNumber';

export default function GitHubStars() {
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    fetch('https://api.github.com/repos/mindverse/Second-Me')
      .then((res) => res.json())
      .then((data) => {
        if (data.stargazers_count !== undefined) {
          setStars(data.stargazers_count);
        }
      })
      .catch((err) => {
        console.error('Failed to fetch stars', err);
      });
  }, []);

  return (
    <a
      className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded transition text-gray-800"
      href="https://github.com/mindverse/Second-Me"
      rel="noreferrer"
      target="_blank"
    >
      <GithubOutlined className="text-lg" />
      {stars !== null && `${formatNumber(stars)}`}
    </a>
  );
}
