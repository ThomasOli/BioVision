import { useState, useCallback } from 'react';

const useUndoRedo = () => {
  const [history, setHistory] = useState<string[]>([]); // Store past states (e.g., canvas data URL)
  const [future, setFuture] = useState<string[]>([]); // Store future states for redo

  const pushToHistory = useCallback((newState: string) => {
    setHistory((prev) => [...prev, newState]);
    setFuture([]); // Clear future when a new action is made
  }, []);

  const undo = useCallback(() => {
    setHistory((prev) => {
      if (prev.length === 0) return prev;

      const newFuture = prev[prev.length - 1];
      setFuture((future) => [newFuture, ...future]);
      return prev.slice(0, -1); // Remove the last state from history
    });
  }, []);

  const redo = useCallback(() => {
    setFuture((prev) => {
      if (prev.length === 0) return prev;

      const newHistory = prev[0];
      setHistory((history) => [...history, newHistory]);
      return prev.slice(1); // Remove the first state from future
    });
  }, []);

  const canUndo = history.length > 0;
  const canRedo = future.length > 0;

  return { history, future, pushToHistory, undo, redo, canUndo, canRedo };
};

export default useUndoRedo;