/**
 * Utility functions for the MemoryLayer SDK
 */

/**
 * Converts camelCase keys to snake_case for API requests
 */
export function toSnakeCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const snakeKey = key.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
    result[snakeKey] = value;
  }
  return result;
}

/**
 * Converts snake_case keys to camelCase for API responses
 */
export function toCamelCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
    result[camelKey] = value;
  }
  return result;
}

/**
 * Sleep for a specified number of milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Auto-paginate a limit/offset list endpoint into an async iterator of items.
 *
 * `fetchPage` is called with `{ limit, offset }` and must return the items for
 * that page. Iteration stops when a page returns fewer items than `pageSize`
 * (the last page) or an empty page. Each item is yielded individually, so
 * callers can `for await (const item of paginate(...))` without managing offsets.
 *
 * The server caps `limit` at 200; pageSize is clamped to that maximum so
 * callers passing a larger value don't receive a 422 validation error.
 *
 * @param fetchPage Function that fetches one page given limit/offset.
 * @param pageSize  Page size to request (default 100, max 200).
 */
export async function* paginate<T>(
  fetchPage: (params: { limit: number; offset: number }) => Promise<T[]>,
  pageSize = 100,
): AsyncGenerator<T> {
  const effectivePageSize = Math.min(pageSize, 200);
  let offset = 0;
  for (;;) {
    const page = await fetchPage({ limit: effectivePageSize, offset });
    if (page.length === 0) {
      return;
    }
    for (const item of page) {
      yield item;
    }
    if (page.length < effectivePageSize) {
      return;
    }
    offset += effectivePageSize;
  }
}

/**
 * Collect all items from an auto-paginating source into a single array.
 * Convenience wrapper around {@link paginate} for callers that want everything
 * eagerly. Use with care on large datasets — prefer the async iterator.
 */
export async function collectPages<T>(
  fetchPage: (params: { limit: number; offset: number }) => Promise<T[]>,
  pageSize = 100,
): Promise<T[]> {
  const out: T[] = [];
  for await (const item of paginate(fetchPage, pageSize)) {
    out.push(item);
  }
  return out;
}

/**
 * Retry a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  initialDelay = 1000
): Promise<T> {
  let lastError: Error | undefined;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < maxRetries - 1) {
        const delay = initialDelay * Math.pow(2, i);
        await sleep(delay);
      }
    }
  }

  throw lastError;
}
