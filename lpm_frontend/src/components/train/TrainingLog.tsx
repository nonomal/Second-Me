import { useEffect, useRef, useState } from 'react';

interface TrainingLogProps {
  trainingDetails: {
    message: string;
    timestamp: string;
  }[];
}

const TrainingLog: React.FC<TrainingLogProps> = ({ trainingDetails }: TrainingLogProps) => {
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const userScrollTimeout = useRef<NodeJS.Timeout | null>(null);

  // Smooth scroll console to bottom
  const smoothScrollConsole = () => {
    if (consoleEndRef.current && !isUserScrolling) {
      const consoleContainer = consoleEndRef.current;

      if (consoleContainer instanceof HTMLElement) {
        consoleContainer.scrollTo({
          top: consoleContainer.scrollHeight,
          behavior: 'smooth'
        });
      }
    }
  };

  useEffect(() => {
    // Set up scroll event listener to detect user scrolling
    const handleUserScroll = () => {
      setIsUserScrolling(true);

      // Clear any existing timeout
      if (userScrollTimeout.current) {
        clearTimeout(userScrollTimeout.current);
      }

      // Reset the flag after a short delay
      userScrollTimeout.current = setTimeout(() => {
        setIsUserScrolling(false);
      }, 2000); // 2 seconds delay before allowing auto-scroll again
    };

    // Find the console container and attach the scroll listener
    if (consoleEndRef.current) {
      const consoleContainer = consoleEndRef.current;

      if (consoleContainer instanceof HTMLElement) {
        consoleContainer.addEventListener('scroll', handleUserScroll);

        // Cleanup function
        return () => {
          consoleContainer.removeEventListener('scroll', handleUserScroll);

          if (userScrollTimeout.current) {
            clearTimeout(userScrollTimeout.current);
          }
        };
      }
    }
  }, []);

  useEffect(() => {
    if (trainingDetails.length > 0) {
      smoothScrollConsole();
    }
  }, [trainingDetails]);

  return (
    <div className="mt-4">
      <h4 className="text-sm font-medium text-gray-700 mb-2">Training Log</h4>
      <div
        ref={consoleEndRef}
        className="bg-gray-900 rounded-lg p-4 h-[600px] overflow-y-auto font-mono text-xs"
      >
        <div className="space-y-1">
          {trainingDetails.length > 0 ? (
            trainingDetails.map((detail, index) => (
              <div key={detail.timestamp + detail.message + index} className="text-gray-300">
                {detail.message}
              </div>
            ))
          ) : (
            <div className="text-gray-300">
              No training logs available. Start training to see logs here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingLog;
