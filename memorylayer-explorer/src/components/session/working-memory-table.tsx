'use client';

import { useState } from 'react';
import { Pencil, Trash2, Plus, Check, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useWorkingMemory, useSetWorkingMemory } from '@/hooks/use-sessions';
import { Skeleton } from '@/components/ui/skeleton';

interface WorkingMemoryTableProps {
  sessionId: string;
}

function formatValue(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

function parseInputValue(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

export function WorkingMemoryTable({ sessionId }: WorkingMemoryTableProps) {
  const { data: memory, isLoading } = useWorkingMemory(sessionId);
  const setWorkingMemory = useSetWorkingMemory();
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');
  const [showAddForm, setShowAddForm] = useState(false);

  if (isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }

  const entries = memory ? Object.entries(memory) : [];

  function startEdit(key: string, value: unknown) {
    setEditingKey(key);
    setEditValue(formatValue(value));
  }

  function cancelEdit() {
    setEditingKey(null);
    setEditValue('');
  }

  function saveEdit(key: string) {
    setWorkingMemory.mutate(
      { sessionId, key, value: parseInputValue(editValue) },
      { onSuccess: () => cancelEdit() }
    );
  }

  function handleAdd() {
    if (!newKey.trim()) return;
    setWorkingMemory.mutate(
      { sessionId, key: newKey.trim(), value: parseInputValue(newValue) },
      {
        onSuccess: () => {
          setNewKey('');
          setNewValue('');
          setShowAddForm(false);
        },
      }
    );
  }

  function handleDelete(key: string) {
    // Setting to null removes the key
    setWorkingMemory.mutate({ sessionId, key, value: null });
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border">
        <table className="w-full">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-4 py-2 text-left text-xs font-medium text-muted-foreground">
                Key
              </th>
              <th className="px-4 py-2 text-left text-xs font-medium text-muted-foreground">
                Value
              </th>
              <th className="w-24 px-4 py-2 text-right text-xs font-medium text-muted-foreground">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {entries.length === 0 && (
              <tr>
                <td colSpan={3} className="px-4 py-8 text-center text-sm text-muted-foreground">
                  No working memory entries
                </td>
              </tr>
            )}
            {entries.map(([key, value]) => (
              <tr key={key} className="border-b last:border-b-0">
                <td className="px-4 py-2 font-mono text-sm">{key}</td>
                <td className="px-4 py-2 text-sm">
                  {editingKey === key ? (
                    <textarea
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      rows={3}
                    />
                  ) : (
                    <pre className="max-w-md truncate whitespace-pre-wrap font-mono text-xs text-muted-foreground">
                      {formatValue(value)}
                    </pre>
                  )}
                </td>
                <td className="px-4 py-2 text-right">
                  {editingKey === key ? (
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => saveEdit(key)}
                        disabled={setWorkingMemory.isPending}
                      >
                        <Check className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={cancelEdit}
                      >
                        <X className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  ) : (
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => startEdit(key, value)}
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-destructive"
                        onClick={() => handleDelete(key)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showAddForm ? (
        <div className="flex items-start gap-2 rounded-lg border p-3">
          <Input
            placeholder="Key"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            className="max-w-[200px] font-mono"
          />
          <textarea
            placeholder="Value (JSON or plain text)"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            className="flex-1 rounded-md border border-input bg-background px-3 py-2 font-mono text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            rows={2}
          />
          <Button size="sm" onClick={handleAdd} disabled={setWorkingMemory.isPending}>
            Add
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setShowAddForm(false);
              setNewKey('');
              setNewValue('');
            }}
          >
            Cancel
          </Button>
        </div>
      ) : (
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowAddForm(true)}
          className="gap-1"
        >
          <Plus className="h-3.5 w-3.5" />
          Add Entry
        </Button>
      )}
    </div>
  );
}
