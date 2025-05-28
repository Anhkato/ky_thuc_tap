<?php

namespace App\Rules;

use Closure;
use Illuminate\Contracts\Validation\ValidationRule;

class NoProfanity implements ValidationRule
{
    /**
     * Run the validation rule.
     *
     * @param  string  $attribute
     * @param  mixed  $value
     * @param  \Closure(string): void  $fail
     * @return void
     */
    public function validate(string $attribute, mixed $value, Closure $fail): void
    {
        $ban = [
            'ditme', 'xam lon', 'con cac', 'dmm', 'dm',
            'vcl', 'vl', 'cc', 'clm', 'clmm', 'clmcc', 'clmmc', 'clmmcc'
        ];

        foreach ($ban as $key) {
            if (stripos($value, $key) !== false) {
                $fail("Trường :attribute chứa từ ngữ không phù hợp.");
                break;
            }
        }
    }
}
