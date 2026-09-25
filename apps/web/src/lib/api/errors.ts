import { unstable_rethrow } from "next/navigation";

/** Serializable error shape shared by server components, server actions and client fetches. */
export interface ApiErrorInfo {
  status: number;
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor({ status, code, message, details }: ApiErrorInfo) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details ?? {};
  }

  toJSON(): ApiErrorInfo {
    return { status: this.status, code: this.code, message: this.message, details: this.details };
  }
}

export const API_UNREACHABLE = "api_unreachable";

export function unreachableError(): ApiError {
  return new ApiError({
    status: 503,
    code: API_UNREACHABLE,
    message: "The Skill Studio API is not reachable. It may be starting up or down.",
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Builds an ApiError from a non-2xx response, reading the `{error: {code, message, details}}` envelope. */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  const envelope = isRecord(body) && isRecord(body.error) ? body.error : undefined;
  return new ApiError({
    status: response.status,
    code: typeof envelope?.code === "string" ? envelope.code : `http_${response.status}`,
    message: typeof envelope?.message === "string" ? envelope.message : response.statusText || "Request failed",
    details: isRecord(envelope?.details) ? envelope.details : undefined,
  });
}

export function toErrorInfo(error: unknown): ApiErrorInfo {
  if (error instanceof ApiError) return error.toJSON();
  return {
    status: 500,
    code: "unexpected",
    message: error instanceof Error ? error.message : "Something went wrong",
  };
}

/** Human readable message for toasts; validation details are summarised, not dumped. */
export function describeError(error: ApiErrorInfo): string {
  if (error.code === API_UNREACHABLE) return error.message;
  if (error.status === 403) return error.message || "You do not have permission to do this.";
  return error.message;
}

export type Result<T> = { ok: true; data: T } | { ok: false; error: ApiErrorInfo };

export async function toResult<T>(promise: Promise<T>): Promise<Result<T>> {
  try {
    return { ok: true, data: await promise };
  } catch (error) {
    // redirect()/notFound() are thrown errors and must reach Next.js.
    unstable_rethrow(error);
    return { ok: false, error: toErrorInfo(error) };
  }
}
