'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  ArrowLeft,
  RefreshCw,
  GitCommitHorizontal,
  Trash2,
  Clock,
  Layers,
  Hash,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { ConfirmDialog } from '@/components/shared/confirm-dialog';
import { TimeAgo } from '@/components/shared/time-ago';
import { formatDate } from '@/lib/format';
import {
  useSession,
  useDeleteSession,
  useTouchSession,
} from '@/hooks/use-sessions';
import { WorkingMemoryTable } from './working-memory-table';
import { CommitDialog } from './commit-dialog';
import { parseISO, differenceInMinutes, isPast } from 'date-fns';

interface SessionDetailProps {
  sessionId: string;
}

export function SessionDetail({ sessionId }: SessionDetailProps) {
  const router = useRouter();
  const { data: session, isLoading, isError, error } = useSession(sessionId);
  const deleteSession = useDeleteSession();
  const touchSession = useTouchSession();
  const [showDelete, setShowDelete] = useState(false);
  const [showCommit, setShowCommit] = useState(false);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (isError || !session) {
    return (
      <div className="space-y-4">
        <Button variant="ghost" onClick={() => router.push('/sessions')} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back to Sessions
        </Button>
        <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center">
          <p className="text-lg font-medium text-foreground">Session Not Found</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {(error as Error)?.message ?? 'This session may have expired or been deleted.'}
          </p>
        </div>
      </div>
    );
  }

  const expiresAt = parseISO(session.expires_at);
  const expired = isPast(expiresAt);
  const minutesLeft = differenceInMinutes(expiresAt, new Date());

  return (
    <>
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={() => router.push('/sessions')} className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-start justify-between">
              <div className="space-y-1">
                <CardTitle className="font-mono text-lg">{sessionId}</CardTitle>
                <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                  <Badge variant="secondary" className="gap-1">
                    <Layers className="h-3 w-3" />
                    {session.workspace_id}
                  </Badge>
                  {session.context_id && (
                    <Badge variant="outline" className="gap-1">
                      <Hash className="h-3 w-3" />
                      {session.context_id}
                    </Badge>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-1"
                  onClick={() => touchSession.mutate({ sessionId })}
                  disabled={touchSession.isPending}
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${touchSession.isPending ? 'animate-spin' : ''}`} />
                  Touch
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-1"
                  onClick={() => setShowCommit(true)}
                >
                  <GitCommitHorizontal className="h-3.5 w-3.5" />
                  Commit
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-1 text-destructive"
                  onClick={() => setShowDelete(true)}
                  disabled={deleteSession.isPending}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div>
                <p className="text-xs text-muted-foreground">Created</p>
                <p className="text-sm font-medium">
                  <TimeAgo date={session.created_at} />
                </p>
                <p className="text-xs text-muted-foreground">{formatDate(session.created_at)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Expires</p>
                {expired ? (
                  <p className="text-sm font-medium text-red-600">Expired</p>
                ) : minutesLeft < 5 ? (
                  <p className="text-sm font-medium text-amber-600">
                    In {minutesLeft}m
                  </p>
                ) : (
                  <p className="text-sm font-medium">In {minutesLeft}m</p>
                )}
                <p className="text-xs text-muted-foreground">{formatDate(session.expires_at)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Working Memory</p>
                <p className="text-sm font-medium">
                  {Object.keys(session.working_memory).length} keys
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Tenant</p>
                <p className="truncate font-mono text-sm font-medium">
                  {session.tenant_id}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Working Memory</CardTitle>
          </CardHeader>
          <CardContent>
            <WorkingMemoryTable sessionId={sessionId} />
          </CardContent>
        </Card>
      </div>

      <ConfirmDialog
        open={showDelete}
        onOpenChange={setShowDelete}
        title="Delete Session"
        description="This will permanently delete this session and all its working memory. This action cannot be undone."
        confirmLabel="Delete"
        destructive
        onConfirm={() => {
          deleteSession.mutate(sessionId, {
            onSuccess: () => router.push('/sessions'),
          });
        }}
      />

      <CommitDialog
        sessionId={sessionId}
        open={showCommit}
        onClose={() => setShowCommit(false)}
      />
    </>
  );
}
