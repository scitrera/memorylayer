import { execSync } from "child_process";
import { basename } from "path";

/**
 * Auto-detect workspace ID from current directory.
 * Priority: git remote origin name > git root folder name > cwd folder name
 */
export function detectWorkspaceId(): string {
  try {
    // Try to get git remote origin URL and extract repo name
    const remoteUrl = execSync("git remote get-url origin 2>/dev/null", { encoding: "utf-8" }).trim();
    if (remoteUrl) {
      // Extract repo name from URL (handles both HTTPS and SSH formats)
      const match = remoteUrl.match(/\/([^/]+?)(\.git)?$/);
      if (match) {
        return match[1].replace(/\.git$/, "");
      }
    }
  } catch {
    // Not a git repo or no remote, continue
  }

  try {
    // Try to get git root directory name
    const gitRoot = execSync("git rev-parse --show-toplevel 2>/dev/null", { encoding: "utf-8" }).trim();
    if (gitRoot) {
      return basename(gitRoot);
    }
  } catch {
    // Not a git repo, continue
  }

  // Fall back to current working directory name
  return basename(process.cwd());
}
