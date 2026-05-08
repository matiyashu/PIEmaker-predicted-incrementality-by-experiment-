"use client";

import { useSearchParams } from "next/navigation";

import { isDemoMode } from "@/lib/demo-mode";

/**
 * Resolves the upload_id for the current page:
 *   - If `?upload_id=...` is in the URL, use it.
 *   - Else if demo mode is active, fall back to the canned `mockupload01`
 *     so demo viewers never have to paste an ID.
 *   - Else returns null — page should render its empty-state instructions.
 *
 * Returns also `isFromDemo` so pages can show a tasteful note (e.g. "viewing
 * mock upload — paste a real upload_id to test against the live backend").
 */
export const MOCK_UPLOAD_ID = "mockupload01";

export function useUploadId(): {
  uploadId: string | null;
  isFromDemo: boolean;
} {
  const params = useSearchParams();
  const fromUrl = params?.get("upload_id") ?? null;
  if (fromUrl) return { uploadId: fromUrl, isFromDemo: false };
  if (isDemoMode()) return { uploadId: MOCK_UPLOAD_ID, isFromDemo: true };
  return { uploadId: null, isFromDemo: false };
}
