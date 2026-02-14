'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Clock, Trash2, RefreshCw, GitCommitHorizontal, Key } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TimeAgo } from '@/components/shared/time-ago';
import { ConfirmDialog } from '@/components/shared/confirm-dialog';
import {
  useDeleteSession,
  useTouchSession,
} from '@/hooks/use-sessions';
import { CommitDialog } from './commit-dialog';
import { parseISO, differenceInMinutes, isPast } from 'date-fns';
import type { Session } from '@/types';

interface SessionCardProps {
  session: Session;
}

export function SessionCard({ session }: SessionCardProps) {
  const router = useRouter();
  const deleteSession = useDeleteSession();
  const touchSession = useTouchSession();
  const [showDelete, setShowDelete] = useState(false);
  const [showCommit, setShowCommit] = useState(false);

  const sessionId = session.id;
  const expiresAt = parseISO(session.expires_at);
  const expired = isPast(expiresAt);
  const minutesLeft = differenceInMinutes(expiresAt, new Date());
  const memoryKeyCount = Object.keys(session.working_memory ?? {}).length;

  return (
    <>
      <Card
        className="cursor-pointer transition-shadow hover:shadow-md"
        onClick={() => router.push(`/sessions/${sessionId}`)}
      >
        <CardContent className="p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1 space-y-2">
              <p className="truncate font-mono text-sm font-medium">
                {sessionId}
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary">{session.workspace_id}</Badge>
                {memoryKeyCount > 0 && (
                  <Badge variant="outline" className="gap-1">
                    <Key className="h-3 w-3" />
                    {memoryKeyCount} key{memoryKeyCount !== 1 ? 's' : ''}
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  <TimeAgo date={session.created_at} />
                </span>
                {expired ? (
                  <span className="font-medium text-red-600">Expired</span>
                ) : minutesLeft < 5 ? (
                  <span className="font-medium text-amber-600">
                    Expires in {minutesLeft}m
                  </span>
                ) : (
                  <span>Expires in {minutesLeft}m</span>
                )}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                title="Touch (extend TTL)"
                onClick={(e) => {
                  e.stopPropagation();
                  touchSession.mutate({ sessionId });
                }}
                disabled={touchSession.isPending}
              >
                <RefreshCw className={`h-4 w-4 ${touchSession.isPending ? 'animate-spin' : ''}`} />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                title="Commit session"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowCommit(true);
                }}
              >
                <GitCommitHorizontal className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-destructive"
                title="Delete session"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowDelete(true);
                }}
                disabled={deleteSession.isPending}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={showDelete}
        onOpenChange={setShowDelete}
        title="Delete Session"
        description="This will permanently delete this session and all its working memory. This action cannot be undone."
        confirmLabel="Delete"
        destructive
        onConfirm={() => deleteSession.mutate(sessionId)}
      />

      <CommitDialog
        sessionId={sessionId}
        open={showCommit}
        onClose={() => setShowCommit(false)}
      />
    </>
  );
}
