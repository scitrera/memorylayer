'use client';

import { useState } from 'react';
import { Plus, Terminal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { EmptyState } from '@/components/shared/empty-state';
import { SessionCard } from '@/components/session/session-card';
import { useSessionIds, useCreateSession } from '@/hooks/use-sessions';
import { useConnection } from '@/providers/connection-provider';

export default function SessionsPage() {
  const sessionIds = useSessionIds();
  const createSession = useCreateSession();
  const { connectionConfig } = useConnection();
  const [showCreate, setShowCreate] = useState(false);
  const [ttl, setTtl] = useState(3600);

  function handleCreate() {
    createSession.mutate(
      {
        workspaceId: connectionConfig.workspaceId,
        ttlSeconds: ttl,
      },
      {
        onSuccess: () => {
          setShowCreate(false);
          setTtl(3600);
        },
      }
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Sessions</h1>
          <p className="text-muted-foreground">
            View and manage active sessions
          </p>
        </div>
        <Button onClick={() => setShowCreate(true)} className="gap-2">
          <Plus className="h-4 w-4" />
          New Session
        </Button>
      </div>

      {sessionIds.length === 0 ? (
        <EmptyState
          icon={Terminal}
          title="No active sessions"
          description="Create a new session to start tracking working memory for your agents."
          action={{
            label: 'New Session',
            onClick: () => setShowCreate(true),
          }}
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {sessionIds.map((id) => (
            <SessionCard key={id} sessionId={id} />
          ))}
        </div>
      )}

      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>New Session</DialogTitle>
            <DialogDescription>
              Create a new session with working memory tracking.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Workspace</label>
              <Input
                value={connectionConfig.workspaceId ?? '_default'}
                disabled
                className="font-mono"
              />
              <p className="text-xs text-muted-foreground">
                Uses the connected workspace
              </p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">TTL (seconds)</label>
              <Input
                type="number"
                min={60}
                max={86400}
                value={ttl}
                onChange={(e) => setTtl(Number(e.target.value))}
              />
              <p className="text-xs text-muted-foreground">
                Session expires after this many seconds of inactivity (default: 3600)
              </p>
            </div>
          </div>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={createSession.isPending}>
              {createSession.isPending ? 'Creating...' : 'Create Session'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
