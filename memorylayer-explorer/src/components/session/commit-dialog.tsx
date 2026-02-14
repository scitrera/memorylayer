'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { useCommitSession } from '@/hooks/use-sessions';
import type { CommitResponse } from '@/types';
import { formatLatency } from '@/lib/format';
import { CheckCircle2 } from 'lucide-react';

interface CommitDialogProps {
  sessionId: string;
  open: boolean;
  onClose: () => void;
}

export function CommitDialog({ sessionId, open, onClose }: CommitDialogProps) {
  const commitSession = useCommitSession();
  const [minImportance, setMinImportance] = useState(0.5);
  const [deduplicate, setDeduplicate] = useState(true);
  const [maxMemories, setMaxMemories] = useState(50);
  const [result, setResult] = useState<CommitResponse | null>(null);

  function handleClose() {
    setResult(null);
    onClose();
  }

  async function handleSubmit() {
    const response = await commitSession.mutateAsync({
      sessionId,
      options: {
        minImportance,
        deduplicate,
        maxMemories,
      },
    });
    setResult(response);
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Commit Session</DialogTitle>
          <DialogDescription>
            Extract and persist working memory into long-term memories.
          </DialogDescription>
        </DialogHeader>

        {result ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-emerald-600">
              <CheckCircle2 className="h-5 w-5" />
              <span className="font-medium">Commit successful</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border p-3">
                <p className="text-xs text-muted-foreground">Extracted</p>
                <p className="text-lg font-semibold">{result.memories_extracted}</p>
              </div>
              <div className="rounded-lg border p-3">
                <p className="text-xs text-muted-foreground">Created</p>
                <p className="text-lg font-semibold">{result.memories_created}</p>
              </div>
              <div className="rounded-lg border p-3">
                <p className="text-xs text-muted-foreground">Deduplicated</p>
                <p className="text-lg font-semibold">{result.memories_deduplicated}</p>
              </div>
              <div className="rounded-lg border p-3">
                <p className="text-xs text-muted-foreground">Time</p>
                <p className="text-lg font-semibold">{formatLatency(result.extraction_time_ms)}</p>
              </div>
            </div>
            <DialogFooter>
              <Button onClick={handleClose}>Close</Button>
            </DialogFooter>
          </div>
        ) : (
          <div className="space-y-5">
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Min Importance: {minImportance.toFixed(2)}
              </label>
              <Slider
                value={[minImportance]}
                onValueChange={([v]) => setMinImportance(v)}
                min={0}
                max={1}
                step={0.05}
              />
            </div>

            <div className="flex items-center gap-3">
              <label className="text-sm font-medium">Deduplicate</label>
              <button
                type="button"
                role="switch"
                aria-checked={deduplicate}
                onClick={() => setDeduplicate(!deduplicate)}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
                  deduplicate ? 'bg-primary' : 'bg-slate-200'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition-transform ${
                    deduplicate ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Max Memories</label>
              <Input
                type="number"
                min={1}
                max={500}
                value={maxMemories}
                onChange={(e) => setMaxMemories(Number(e.target.value))}
              />
            </div>

            <DialogFooter className="gap-2 sm:gap-0">
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                onClick={handleSubmit}
                disabled={commitSession.isPending}
              >
                {commitSession.isPending ? 'Committing...' : 'Commit'}
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
