<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;
use App\Rules\NoProfanity;

class StoreCandidateRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'name' => 'required|min:5',
            'email' => 'required|email|unique:candidates,email',
            'birthday' => 'required|date|before:-18 years',
            'avatar' => 'nullable|image|max:2048',
            'cv' => 'required|file|mimetypes:application/pdf|max:5120',
            'mota' => ['nullable', 'max:1000', new NoProfanity()],
        ];
    }
}
